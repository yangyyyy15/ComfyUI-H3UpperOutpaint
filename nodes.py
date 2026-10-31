"""Spatial source locking for H3; sampling and compositing remain native nodes."""

import json
import math

import torch
import torch.nn.functional as F

from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_minimax_h3 import temporal_shape


class H3UpperOutpaintResolutionPlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "source_width": ("INT", {"default": 720, "min": 32, "max": 16384, "step": 1}),
            "source_height": ("INT", {"default": 1280, "min": 32, "max": 16384, "step": 1}),
            "upper_headroom_ratio": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 2.0, "step": 0.01,
                "tooltip": "Planned upper area / mapped source height: 0.25 adds 25%, 0.50 adds 50%. Apply planned dimensions when outpainting the Qwen anchor."}),
            "minimum_upper_pixels": ("INT", {"default": 64, "min": 32, "max": 4096, "step": 32,
                "tooltip": "Minimum upper area in the planned canvas. An existing anchor below this target is allowed, with a warning."}),
            "longest_edge": ("INT", {"default": 768, "min": 128, "max": 4096, "step": 32,
                "tooltip": "Maximum planned canvas edge, including the added upper region. Does not resize an existing anchor."}),
        }, "optional": {"anchor_image": ("IMAGE", {"tooltip": "FINAL resized anchor. width/height follow this image; planned_* outputs describe the next anchor to create."})}}

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING", "INT", "INT", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("width", "height", "mapped_source_height", "upper_pixels", "report",
                    "planned_width", "planned_height", "planned_source_height", "planned_upper_pixels", "is_source_portrait")
    FUNCTION = "plan"
    CATEGORY = "MiniMax H3/Outpaint"

    def plan(self, source_width, source_height, anchor_image=None, upper_headroom_ratio=0.25,
             minimum_upper_pixels=64, longest_edge=768):
        if source_width <= 0 or source_height <= 0:
            raise ValueError("Source dimensions must be positive.")
        if not math.isfinite(upper_headroom_ratio) or upper_headroom_ratio < 0:
            raise ValueError("upper_headroom_ratio must be finite and nonnegative.")
        if minimum_upper_pixels < 32 or longest_edge < 32 or minimum_upper_pixels % 32 or longest_edge % 32:
            raise ValueError("minimum_upper_pixels and longest_edge must be positive multiples of 32.")

        def geometry_at_width(w):
            mapped = max(32, round(w * source_height / source_width / 32) * 32)
            upper = max(minimum_upper_pixels, math.ceil(mapped * upper_headroom_ratio / 32) * 32)
            return w, mapped + upper, mapped, upper

        # Round source and extension separately, then fit the complete canvas to the budget.
        planned = None
        for candidate_width in range(longest_edge, 31, -32):
            candidate = geometry_at_width(candidate_width)
            if candidate[1] <= longest_edge:
                planned = candidate
                break
        warnings = []
        if planned is None:
            # Keep a usable anchor executable even if its next-image planning budget is impossible.
            planned = geometry_at_width(32)
            if anchor_image is None:
                raise ValueError(
                    f"No 32px-grid canvas fits longest_edge={longest_edge}. "
                    f"Smallest plan is {planned[0]}x{planned[1]}; increase longest_edge "
                    "or reduce minimum_upper_pixels / upper_headroom_ratio.")
            warnings.append(f"Planned canvas exceeds longest_edge={longest_edge}; increase the planning budget.")
        if anchor_image is not None:
            if anchor_image.ndim != 4 or anchor_image.shape[0] < 1:
                raise ValueError("anchor_image must be an IMAGE tensor.")
            height, width = anchor_image.shape[1:3]
            if width % 32 or height % 32:
                raise ValueError("Connect the final resized anchor with dimensions divisible by 32.")
            mapped = max(32, round(width * source_height / source_width / 32) * 32)
            upper = height - mapped
            if upper < 32:
                needed = geometry_at_width(width)
                raise ValueError(
                    f"Anchor {width}x{height} leaves {upper}px above mapped source height {mapped}px. "
                    f"At this width, outpaint to {needed[0]}x{needed[1]} for the requested headroom. "
                    "Keep source scale and bottom alignment; resizing alone does not add content.")
            requested_upper = geometry_at_width(width)[3]
            if upper < requested_upper:
                warnings.append(
                    f"Existing anchor has {upper}px above the source; requested {requested_upper}px. "
                    f"It will run unchanged. To apply this expansion at the same source scale, "
                    f"outpaint the anchor to {width}x{mapped + requested_upper}.")
        else:
            width, height, mapped, upper = planned
        target_ratio = source_width / (source_height * (1.0 + upper_headroom_ratio))
        report = json.dumps({
            "source": [source_width, source_height],
            "target_canvas": [width, height],
            "anchor_input": list(anchor_image.shape[1:3]) if anchor_image is not None else None,
            "mapped_source_height": mapped,
            "upper_pixels": upper,
            "mode": "read_final_anchor" if anchor_image is not None else "plan_dimensions",
            "warnings": warnings,
            "planned_canvas": list(planned[:2]),
            "planned_source_height": planned[2],
            "planned_upper_pixels": planned[3],
            "requested_upper_headroom_ratio": upper_headroom_ratio,
            "anchor_needed_at_current_width": list(geometry_at_width(width)[:2]),
            "upper_headroom_ratio": round(upper / mapped, 4),
            "longest_edge": longest_edge,
            "grid": 32,
            "ratio_error": round(abs(width / height - target_ratio), 6),
            "note": "Use width/height for H3 conditioning and planned_width/planned_height for the next Qwen anchor. "
                    "Keep the source at the planned size, bottom aligned, and generate the upper region. "
                    "Changing the plan does not change an existing image or guarantee full-head visibility."
        }, indent=2)
        return width, height, mapped, upper, report, *planned, source_height > source_width


class H3UpperOutpaintTopPadding:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "source_width": ("INT", {"default": 1920, "min": 1, "max": 16384,
                "forceInput": True, "tooltip": "ImageSizeInfo width, measured immediately BEFORE padding."}),
            "source_height": ("INT", {"default": 1080, "min": 1, "max": 16384,
                "forceInput": True, "tooltip": "ImageSizeInfo height. This image must have the original video's aspect ratio."}),
            "upper_headroom_ratio": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 2.0, "step": 0.01,
                "tooltip": "Top expansion relative to source height: 0.50 means at least 50%."}),
            "video_longest_edge": ("INT", {"default": 1280, "min": 128, "max": 4096, "step": 32,
                "tooltip": "Match the VIDEO workflow's anchor scaler: original ratio, longest edge, grid 32."}),
            "minimum_upper_pixels": ("INT", {"default": 128, "min": 32, "max": 4096, "step": 32,
                "tooltip": "Minimum upper area AFTER video resizing. Match or exceed the video planner's minimum."}),
        }}

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT", "INT", "STRING", "BOOLEAN")
    RETURN_NAMES = ("top", "padded_width", "padded_height", "video_width", "video_height",
                    "mapped_source_height", "video_upper_pixels", "report", "is_source_portrait")
    FUNCTION = "calculate"
    CATEGORY = "MiniMax H3/Outpaint"

    def calculate(self, source_width, source_height, upper_headroom_ratio=0.25,
                  video_longest_edge=1280, minimum_upper_pixels=128):
        if source_width <= 0 or source_height <= 0:
            raise ValueError("Connect positive ImageSizeInfo width/height from the image BEFORE padding.")
        if not math.isfinite(upper_headroom_ratio) or upper_headroom_ratio < 0:
            raise ValueError("upper_headroom_ratio must be finite and nonnegative.")
        if video_longest_edge < 64 or video_longest_edge % 32:
            raise ValueError("video_longest_edge must be a multiple of 32, at least 64.")
        if minimum_upper_pixels < 32 or minimum_upper_pixels % 32:
            raise ValueError("minimum_upper_pixels must be a positive multiple of 32.")
        if minimum_upper_pixels > video_longest_edge - 32:
            raise ValueError("minimum_upper_pixels leaves no room for the source. Increase video_longest_edge.")

        # KJ padding uses 8px steps. Predict LayerStyle's original/longest/grid-32 resize,
        # then use the SAME source-height rounding as H3UpperOutpaintPrepare.
        first_top = max(8, math.ceil(source_height * upper_headroom_ratio / 8) * 8)
        for top in range(first_top, 16385, 8):
            padded_height = source_height + top
            ratio = source_width / padded_height
            if ratio > 1:
                width, height = video_longest_edge, int(video_longest_edge / ratio)
            else:
                width, height = int(video_longest_edge * ratio), video_longest_edge
            width = max(32, (width + 31) // 32 * 32)
            height = max(32, (height + 31) // 32 * 32)
            mapped = max(32, round(width * source_height / source_width / 32) * 32)
            upper = height - mapped
            required = max(minimum_upper_pixels, math.ceil(mapped * upper_headroom_ratio / 32) * 32)
            if upper >= required:
                break
        else:
            raise ValueError(
                "Required top padding exceeds KJ's 16384px input limit. Resize the source image first "
                "while preserving its aspect ratio, increase video_longest_edge or reduce the expansion target.")

        report = json.dumps({
            "source": [source_width, source_height],
            "padding": {"left": 0, "top": top, "right": 0, "bottom": 0},
            "padded_canvas": [source_width, padded_height],
            "requested_upper_headroom_ratio": upper_headroom_ratio,
            "raw_upper_headroom_ratio": round(top / source_height, 4),
            "expected_video_canvas": [width, height],
            "mapped_source_height": mapped,
            "video_upper_pixels": upper,
            "required_video_upper_pixels": required,
            "video_longest_edge": video_longest_edge,
            "expected_geometry_check": "pass",
            "note": "Connect top to Image Pad For Outpaint Masked; keep left/right/bottom at 0. "
                    "Run Qwen to fill the padded area and preserve the padded canvas and source placement. "
                    "Prediction assumes the same source aspect ratio and LayerStyle original/longest/grid-32 "
                    "settings downstream. It verifies dimensions, not generated head visibility or image quality."
        }, indent=2)
        return top, source_width, padded_height, width, height, mapped, upper, report, source_height > source_width


class H3UpperOutpaintPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "av_latent": ("LATENT",),
            "video_vae": ("VAE",),
            "source_frames": ("IMAGE",),
            "expanded_frame": ("IMAGE",),
            "source_height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 32,
                "tooltip": "0: fit source to canvas width, round height to 32. Otherwise specify its height inside the anchor."}),
            "seam_pixels": ("INT", {"default": 64, "min": 32, "max": 1024, "step": 32,
                "tooltip": "Source's upper rows released for regeneration, aligned to H3 patches."}),
            "feather_pixels": ("INT", {"default": 32, "min": 0, "max": 1024, "step": 1}),
            "source_pixels": (["blend_seam", "strict_source"],),
        }}

    RETURN_TYPES = ("LATENT", "IMAGE", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("masked_av_latent", "source_canvas", "source_keep_mask", "repaint_mask", "report")
    FUNCTION = "prepare"
    CATEGORY = "MiniMax H3/Outpaint"

    def prepare(self, av_latent, video_vae, source_frames, expanded_frame,
                source_height=0, seam_pixels=64, feather_pixels=32, source_pixels="blend_seam"):
        if not isinstance(av_latent.get("samples"), NestedTensor):
            raise ValueError("Connect H3 joint AV latent, not a plain image/video latent.")
        video, audio = av_latent["samples"].unbind()
        if video.ndim != 5 or video.shape[:2] != (1, 24):
            raise ValueError("Expected H3 video latent [1,24,T,H/16,W/16].")
        if expanded_frame.shape[0] != 1 or expanded_frame.shape[-1] != 3:
            raise ValueError("expanded_frame must be one RGB image, matching the first_frame input.")
        count, raw_h, raw_w, channels = source_frames.shape
        height, width = expanded_frame.shape[1:3]
        if channels != 3 or height % 32 or width % 32:
            raise ValueError("Use RGB frames and an expanded canvas with width/height divisible by 32.")
        if tuple(video.shape[-2:]) != (height // 16, width // 16):
            raise ValueError("H3 latent canvas differs from expanded_frame. Connect its size to conditioning.")
        aligned_count, latent_t, audio_t = temporal_shape(count)
        if count != aligned_count or video.shape[2] != latent_t or audio.shape[-1] != audio_t:
            raise ValueError(f"Source has {count} frames; use an H3 length 17*n+5 (e.g. 73 or 124), "
                             "and set the same length in VHS and H3 conditioning. No frames are repeated silently.")
        aspect_h = max(32, round(raw_h * width / raw_w / 32) * 32)
        fitted_h = source_height or aspect_h
        top = height - fitted_h
        if fitted_h % 32 or top < 32:
            raise ValueError(
                f"No usable upper expansion area: source={raw_w}x{raw_h}, "
                f"anchor={width}x{height}, aspect-preserving source_on_canvas={width}x{aspect_h}, "
                f"computed_top={top}px. The anchor must be taller than the source "
                "after width alignment. For a 720x1280 source at 1152px width, "
                "the mapped source height is 2048px; a 1152x2240 anchor leaves 192px above it. "
                "source_height must be a 32-pixel multiple and must not distort the source aspect.")
        if source_height and abs(fitted_h - aspect_h) > max(32, round(aspect_h * 0.02)):
            raise ValueError(
                f"source_height={source_height} would distort source aspect: expected about {aspect_h}px "
                f"for source={raw_w}x{raw_h} on anchor width {width}px. Create a taller anchor instead.")
        if seam_pixels % 32 or seam_pixels < 32:
            raise ValueError("seam_pixels must be a positive multiple of 32.")
        if not 0 <= feather_pixels <= seam_pixels:
            raise ValueError("feather_pixels must be between 0 and seam_pixels.")
        requested_seam, requested_feather = seam_pixels, feather_pixels
        # Small/landscape sources must retain at least one locked source patch.
        seam_pixels = min(seam_pixels, max(0, fitted_h - 32))
        feather_pixels = min(feather_pixels, seam_pixels)
        if source_pixels not in ("blend_seam", "strict_source"):
            raise ValueError("Unknown source_pixels mode.")

        source = F.interpolate(source_frames.movedim(-1, 1), (fitted_h, width),
                               mode="bilinear", align_corners=False, antialias=True).movedim(1, -1)
        # Encode only actual source pixels. A black padded canvas must never be pinned.
        encoded = video_vae.encode(source)
        expected = (1, 24, latent_t, fitted_h // 16, width // 16)
        if tuple(encoded.shape) != expected:
            raise ValueError(f"Video VAE returned {tuple(encoded.shape)}; expected {expected}. Use H3 video VAE.")
        generated = torch.zeros_like(video)
        generated[..., top // 16:, :] = encoded.to(device=video.device, dtype=video.dtype)
        video_mask = video.new_ones((1, 1, latent_t, height // 16, width // 16))
        video_mask[..., (top + seam_pixels) // 16:, :] = 0
        masks = av_latent.get("noise_mask")
        if masks is None:
            audio_mask = torch.ones_like(audio)
        elif isinstance(masks, NestedTensor):
            original_video_mask, audio_mask = masks.unbind()
            if not bool(torch.all(original_video_mask == 1)):
                raise ValueError("Connect the original H3 conditioning latent; an existing spatial mask would be overwritten.")
            audio_mask = audio_mask.clone()
        else:
            raise ValueError("H3 AV noise_mask must contain separate video/audio masks.")
        output = av_latent.copy()
        output["samples"] = NestedTensor((generated, audio))
        output["noise_mask"] = NestedTensor((video_mask, audio_mask))

        # Only the keep mask can copy these pixels into the final decoded video.
        canvas = expanded_frame.to(source).expand(count, -1, -1, -1).clone()
        canvas[:, top:] = source
        keep = source.new_zeros((1, height, width))
        if source_pixels == "strict_source":
            keep[:, top:] = 1
        else:
            keep[:, top + seam_pixels:] = 1
            if feather_pixels:
                fade = torch.linspace(0, 1, feather_pixels + 2, device=source.device, dtype=source.dtype)[1:-1]
                keep[:, top + seam_pixels - feather_pixels:top + seam_pixels] = fade[None, :, None]
        repaint = source.new_ones((1, height, width))
        repaint[:, top + seam_pixels:] = 0
        anchor_error = (source[:1, seam_pixels:] - expanded_frame.to(source)[:, top + seam_pixels:]).abs().mean().item()
        report = json.dumps({
            "experimental": True, "canvas": [width, height], "source": [width, fitted_h],
            "source_top": top, "frames": count, "fps_required": 24,
            "fully_sampled_rows": [0, top + seam_pixels],
            "locked_rows": [top + seam_pixels, height], "source_pixels": source_pixels,
            "requested_seam_pixels": requested_seam, "seam_pixels": seam_pixels,
            "requested_feather_pixels": requested_feather,
            "feather_pixels": feather_pixels, "audio_mask_preserved": True,
            "warning": "Seam capped to retain 32px of source; inspect how much body is regenerated."
                       if requested_seam != seam_pixels else None,
            "anchor_source_mae": round(anchor_error, 6),
            "note": "Check anchor/source alignment in preview; Qwen must preserve source scale and placement. "
                    "Prompt and masks do not guarantee anatomy or temporal quality. Source pixels are exact only "
                    "after resize and before lossy video encoding, outside the seam in blend_seam mode."
        }, indent=2)
        return output, canvas, keep, repaint, report
