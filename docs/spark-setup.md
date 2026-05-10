# Running the basketball tracker's inference on a DGX Spark

This is a one-pager for someone hosting the inference container on their NVIDIA DGX Spark. The pipeline (which runs elsewhere — laptop, server, whatever) will POST frames to the Spark and get back JSON detections.

## Why this works without code changes

The pipeline already speaks the **Roboflow inference HTTP protocol**. Roboflow ships a Docker image (`roboflow/roboflow-inference-server-gpu`) that exposes that exact protocol on port 9001 and runs the same models locally on your GPU. The pipeline just points at your Spark instead of `detect.roboflow.com`. ~20-50× faster than the hosted API.

## What you need

- DGX Spark, powered on, connected to the network
- A Roboflow API key (the *consumer* of the service supplies their own — you don't need yours)
- Docker installed (preinstalled on Spark)
- Tailscale (recommended) so the laptop side can reach the Spark across networks

## Run the inference server

```bash
# On the Spark
export ROBOFLOW_API_KEY=<your_key>   # used to download model weights once

docker run -d --name roboflow-inference \
  --gpus all \
  -p 9001:9001 \
  -e ROBOFLOW_API_KEY=$ROBOFLOW_API_KEY \
  --restart unless-stopped \
  roboflow/roboflow-inference-server-gpu:latest
```

**If `:latest` errors with "no kernel image available"** — that's the GB10/Blackwell SM_121 issue. Use the Jetson-tagged image instead, which uses the same JetPack 7 / CUDA 13 stack that's confirmed working on Spark:

```bash
docker run -d --name roboflow-inference \
  --gpus all -p 9001:9001 \
  -e ROBOFLOW_API_KEY=$ROBOFLOW_API_KEY \
  --restart unless-stopped \
  roboflow/roboflow-inference-server-jetson-6.0.0:latest
```

## Verify it's up

```bash
curl http://localhost:9001/
# → {"name":"Roboflow Inference Server", ...}
```

OpenAPI / Swagger UI at `http://localhost:9001/docs`.

## Expose it to the laptop

**Easiest: Tailscale.** Install on the Spark and on the laptop, both join the same tailnet, then the laptop can hit `http://<spark-tailnet-name>:9001` from anywhere.

```bash
# On the Spark
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
tailscale status   # note the magic DNS name, e.g. spark.tail-xxxx.ts.net
```

**Same Wi-Fi only:** Skip Tailscale. Use the Spark's LAN IP directly: `http://192.168.x.y:9001`.

## What to send the laptop side

Two things:
1. The URL: `http://<spark-tailnet-name>:9001` (or LAN IP)
2. Confirm both basketball models are accessible by hitting them once — the server downloads weights on first request:
   ```bash
   curl -X POST "http://localhost:9001/basketball-player-detection-3-ycjdo/6?api_key=$ROBOFLOW_API_KEY&confidence=0.4" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     --data-binary @<(base64 < some_frame.jpg)
   curl -X POST "http://localhost:9001/basketball-court-detection-2/13?api_key=$ROBOFLOW_API_KEY&confidence=0.3" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     --data-binary @<(base64 < some_frame.jpg)
   ```
   Both should return JSON with `predictions`. First call per model takes ~30s (download + compile); after that ~30 ms.

## Troubleshooting

- `nvidia-smi` should show the GB10 from inside the container: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`
- Container logs: `docker logs -f roboflow-inference`
- Port not reachable from laptop: check Spark firewall / Tailscale ACL
- Slow inference: confirm GPU acceleration with `docker logs roboflow-inference | grep -i cuda` — should mention CUDAExecutionProvider, not CPUExecutionProvider
