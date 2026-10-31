"""Video save node with an optional VideoHelperSuite player preview."""


def _get_vhs_video_combine():
    import nodes as comfy_nodes

    registered = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}).get("VHS_VideoCombine")
    if registered is None:
        raise RuntimeError("H3 Outpaint Video Save requires the VideoHelperSuite VHS_VideoCombine node.")
    return registered


class H3UpperOutpaintVideoCombine:
    @classmethod
    def INPUT_TYPES(cls):
        schema = _get_vhs_video_combine().INPUT_TYPES()
        required = dict(schema.get("required", {}))
        optional = dict(schema.get("optional", {}))
        hidden = dict(schema.get("hidden", {}))
        required["images"] = ("IMAGE",)
        required["frame_rate"] = ("FLOAT", {"default": 24.0, "min": 1.0, "step": 0.01})
        required.setdefault("loop_count", ("INT", {"default": 0, "min": 0, "max": 100}))
        required.setdefault("filename_prefix", ("STRING", {"default": "H3Outpaint"}))
        required.setdefault("format", (["video/h264-mp4", "video/webm", "image/gif"],
                                        {"default": "video/h264-mp4"}))
        required.setdefault("pingpong", ("BOOLEAN", {"default": False}))
        required.setdefault("save_output", ("BOOLEAN", {"default": True}))
        required["show_preview"] = (
            "BOOLEAN", {"default": True, "tooltip": "Show or hide the encoded video player in this node."})
        return {"required": required, "optional": optional, "hidden": hidden}

    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    FUNCTION = "combine_video"
    CATEGORY = "MiniMax H3/Outpaint"
    OUTPUT_NODE = True

    @classmethod
    def combine_video(cls, images, frame_rate, loop_count=0, filename_prefix="H3Outpaint",
                      format="video/h264-mp4", pingpong=False, save_output=True,
                      show_preview=True, **kwargs):
        result = _get_vhs_video_combine()().combine_video(
            images=images,
            frame_rate=float(frame_rate),
            loop_count=int(loop_count),
            filename_prefix=filename_prefix,
            format=format,
            pingpong=bool(pingpong),
            save_output=bool(save_output),
            **kwargs,
        )
        ui = result.get("ui", {}) if show_preview else {}
        return {"ui": ui, "result": result.get("result", ((save_output, []),))}


class TwoPersonMotionVideoCombineT8(H3UpperOutpaintVideoCombine):
    """Compatibility node name for workflows saved before the merge."""


NODE_CLASS_MAPPINGS = {
    "H3UpperOutpaintVideoCombine": H3UpperOutpaintVideoCombine,
    "TwoPersonMotionVideoCombineT8": TwoPersonMotionVideoCombineT8,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3UpperOutpaintVideoCombine": "H3 Outpaint Video Save + Preview",
    "TwoPersonMotionVideoCombineT8": "TwoPersonMotion Video Combine (compatibility)",
}
