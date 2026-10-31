from .nodes import (
    H3UpperOutpaintPrepare,
    H3UpperOutpaintResolutionPlanner,
    H3UpperOutpaintTopPadding,
)
from .video_combine import H3UpperOutpaintVideoCombine, TwoPersonMotionVideoCombineT8

NODE_CLASS_MAPPINGS = {
    "H3UpperOutpaintPrepare": H3UpperOutpaintPrepare,
    "H3UpperOutpaintResolutionPlanner": H3UpperOutpaintResolutionPlanner,
    "H3UpperOutpaintTopPadding": H3UpperOutpaintTopPadding,
    "H3UpperOutpaintVideoCombine": H3UpperOutpaintVideoCombine,
    "TwoPersonMotionVideoCombineT8": TwoPersonMotionVideoCombineT8,
}
NODE_DISPLAY_NAME_MAPPINGS = {"H3UpperOutpaintPrepare": "H3 Upper Outpaint Prepare (Experimental)"}
NODE_DISPLAY_NAME_MAPPINGS["H3UpperOutpaintResolutionPlanner"] = "H3 Upper Outpaint Resolution Planner"
NODE_DISPLAY_NAME_MAPPINGS["H3UpperOutpaintTopPadding"] = "H3 Qwen Top Padding Calculator"
NODE_DISPLAY_NAME_MAPPINGS["H3UpperOutpaintVideoCombine"] = "H3 Outpaint Video Save + Preview"
NODE_DISPLAY_NAME_MAPPINGS["TwoPersonMotionVideoCombineT8"] = "TwoPersonMotion Video Combine (compatibility)"
