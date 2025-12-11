# RunPod serverless: WAN2.2 I2V with DiffSynth

This worker loads the WAN2.2 image-to-video 14B model via DiffSynth and serves it as a RunPod serverless endpoint.

## Inputs
- `prompt` (string, required): text prompt.
- `image` (string, required): reference image. Supports URL, base64 (optionally `data:`), or local path.
- `negative_prompt` (string, optional)
- `height` / `width` (int, optional, default 480/832): resize target.
- `tiled` (bool, optional, default true): enable tiling to reduce VRAM.
- `fps` (int, optional, default 15) and `quality` (int, optional, default 5) for output.
- `seed` (int, optional, default 1)
- `control_video` (string, optional): URL or local path to a control/depth video (uses VACE control).

## Outputs
- `output_path`: local mp4 path inside the pod.
- `s3_url`: populated when `BUCKET_NAME`, `ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, and `AWS_SECRET_ACCESS_KEY` are set.

## Running locally
```bash
pip install -r requirements.txt
python handler.py --rp_serve_api --rp_api_port 8000
```
Then POST a job to `http://localhost:8000` with the input payload.

## Notes
- Use a RunPod PyTorch GPU base image so torch with CUDA is preinstalled; `diffsynth` honors existing torch.
- Adjust `WAN_MODEL_ID` and `WAN_TOKENIZER_ID` env vars to swap model variants.
