# RunPod serverless: WAN2.2 I2V with DiffSynth

This worker loads the WAN2.2 image-to-video 14B model via DiffSynth and serves it as a RunPod serverless endpoint.

## Prereqs
- Use a RunPod PyTorch GPU base image so CUDA and torch are preinstalled; torch is intentionally not listed in `requirements.txt`.
- Outbound internet is required on first run to download Wan2.2 weights and tokenizer.
- GPU memory: pipeline uses bfloat16 and CPU offload; limit height/width to sane values (defaults 480x832).

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

## Required env vars
- `WAN_MODEL_ID` (default `PAI/Wan2.2-I2V-A14B`)
- `WAN_TOKENIZER_ID` (default `Wan-AI/Wan2.1-T2V-1.3B`)
- S3 upload (optional): `BUCKET_NAME`, `ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

## Running locally
```bash
pip install -r requirements.txt
python handler.py --rp_serve_api --rp_api_port 8000
```
Then POST a job to `http://localhost:8000` with the input payload.

### Quick smoke test (no S3 upload)
Use a tiny generated image (1x1 white) as base64:
```bash
python - <<'PY'
from PIL import Image
import base64, io, json, requests
buf = io.BytesIO()
Image.new("RGB", (1, 1), (255, 255, 255)).save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode()
payload = {"input": {"prompt": "a short test clip", "image": b64}}
print(json.dumps(payload))
PY
```
Paste the printed JSON into your POST body. Expect an mp4 path in `output_path`.

## Notes
- Use a RunPod PyTorch GPU base image so torch with CUDA is preinstalled; `diffsynth` honors existing torch.
- Adjust `WAN_MODEL_ID` and `WAN_TOKENIZER_ID` env vars to swap model variants.

