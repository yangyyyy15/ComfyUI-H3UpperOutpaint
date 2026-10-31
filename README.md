# ComfyUI-H3UpperOutpaint

ComfyUI custom nodes for MiniMax H3 upper outpainting workflows, plus a VHS-backed video saver with an optional embedded preview.

## Nodes

- `H3UpperOutpaintResolutionPlanner` reports the current anchor geometry and plans a 32-pixel-aligned canvas with configurable upper headroom. It does not generate or resize images.
- `H3UpperOutpaintTopPadding` calculates the top padding needed before Qwen outpainting to meet a target headroom after video resizing.
- `H3UpperOutpaintPrepare` prepares the H3 AV latent and masks for upper-region sampling while preserving the source region and audio mask.
- `H3UpperOutpaintVideoCombine` delegates encoding and saving to VideoHelperSuite and adds a `show_preview` toggle. Disabling the toggle hides the embedded player while keeping the save result.

The legacy node identifier `TwoPersonMotionVideoCombineT8` remains registered so existing workflows can resolve it.

## Install

Install this repository in `ComfyUI/custom_nodes/ComfyUI-H3UpperOutpaint`, install [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite), and restart ComfyUI.

```powershell
cd ComfyUI/custom_nodes
git clone https://github.com/yangyyyy15/ComfyUI-H3UpperOutpaint.git
```

`H3UpperOutpaintPrepare` requires a ComfyUI build that includes the MiniMax H3 AV latent implementation. The other geometry nodes only calculate dimensions; they do not run Qwen or H3 models.

## Notes

The outpainting preparation node is experimental. Geometry checks cannot guarantee generated anatomy, identity, temporal consistency, or visual quality. Inspect outputs before use.
