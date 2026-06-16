# Realtime Deployment Runbook

This runbook records the deployment shape used for the internal 3090/4090 realtime tests. Replace hostnames, ports, model paths, and credentials for other environments. Do not commit API keys or passwords.

## Runtime Versions

Verified model runtime:

```text
Python: 3.10
Torch: 2.7.0+cu126
CUDA runtime in torch: 12.6
Transformers: 4.57.3
Hugging Face Hub: 0.36.0
Tokenizers: 0.22.2
Safetensors: 0.5.3
```

The realtime WebSocket server is implemented by:

```text
examples/realtime_ws_server.py
```

## Recommended 4090 Startup

Use one process per concurrent realtime TTS session. Each process loads its own model instance.

```bash
cd /home/yangr/qwen3-tts-poc/qwen3tts-streaming-9028b0e

CUDA_VISIBLE_DEVICES=0 nohup /home/yangr/qwen3-tts-poc/env-qwen3tts/bin/python examples/realtime_ws_server.py \
  --model /home/yangr/qwen3-tts-poc/models/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --mode custom \
  --device cuda:0 \
  --dtype bfloat16 \
  --speaker Vivian \
  --language Auto \
  --host 0.0.0.0 \
  --port 7861 \
  --emit-every-frames 16 \
  --decode-window-frames 64 \
  --overlap-samples 256 \
  --first-chunk-emit-every 1 \
  --first-chunk-decode-window 8 \
  --first-chunk-frames 2 \
  --initial-buffer-chars 5 \
  --fast-codebook \
  --codebook-cuda-graph \
  --codebook-graph-warmup-runs 3 \
  --no-flash-attn \
  > /home/yangr/qwen3-tts-poc/logs/realtime_ws_7861.log 2>&1 &

echo $! > /home/yangr/qwen3-tts-poc/realtime_ws_7861.pid
```

Start additional instances by changing `--port`, log path, and pid path. Example: `7862`.

## Health Checks

```bash
ss -ltnp | grep 7861
tail -100 /home/yangr/qwen3-tts-poc/logs/realtime_ws_7861.log
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
```

Expected log markers:

```text
Codebook CUDA graph captured successfully
Realtime warmup completed
Uvicorn running on http://0.0.0.0:<port>
```

## Stop / Restart

```bash
kill $(cat /home/yangr/qwen3-tts-poc/realtime_ws_7861.pid)
kill $(cat /home/yangr/qwen3-tts-poc/realtime_ws_7862.pid)
```

Confirm that only the intended PIDs are stopped:

```bash
ps -p $(cat /home/yangr/qwen3-tts-poc/realtime_ws_7861.pid) -o pid,stat,cmd
nvidia-smi
```

## Measured Internal Results

Same text, same client script, same realtime server parameters:

| Host | GPU | Port | Flash-Attn | Client first audio | Server first audio | RTF |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `172.16.99.32` | RTX 3090 24GB | `7861` | installed/enabled | 436 ms | 276.7 ms | 0.895 |
| `172.16.99.91` | RTX 4090 | `7861` | disabled with `--no-flash-attn` | 309 ms | 150.4 ms | 0.666 |
| `172.16.99.91` | RTX 4090 | `7863` test | enabled | 302-322 ms warm | 144-148 ms warm | 0.622-0.670 |

The first flash-attn request on the 4090 test instance showed one-time initialization jitter. Do not use the first request after startup as the only benchmark sample.

## Concurrency Findings

Single process behavior:

```text
two clients -> one model process -> serialized generation
```

Observed result on one process: the second client's first audio arrived only after the first request finished.

Multi-process behavior:

```text
client A -> :7861
client B -> :7862
```

Observed result: both clients received first audio at roughly the same time. This is the currently recommended safe concurrency model.

Resource estimate from the internal 4090 host:

```text
Production baseline GPU memory: about 21 GiB
One TTS instance increment: about 3.0-3.5 GiB
Two TTS instances increment: about 5.6 GiB total in the observed run
```

Keep a safety margin for production processes. Do not fill all remaining VRAM just because the model instances fit.

## Speaker List

The CustomVoice model exposes these built-in speakers:

```text
Vivian
Serena
Uncle_Fu
Dylan
Eric
Ryan
Aiden
Ono_Anna
Sohee
```

Base voice clone can be tested with authorized reference audio. For realtime appendable text, use `x_vector_only_mode=True`; ICL/ref-code prompting is not appendable in the current implementation.

## Node Web Demo Deployment

The separate Node demo can route browser requests to multiple TTS providers:

```text
local 4090 -> ws://172.16.99.91:7861/api-ws/v1/realtime
local 4090 -> ws://172.16.99.91:7862/api-ws/v1/realtime
local 3090 -> ws://172.16.99.32:7861/api-ws/v1/realtime
Aliyun -> wss://dashscope.aliyuncs.com/api-ws/v1/realtime
```

The demo must run on a modern Node runtime. Node 12 is too old because the server uses ESM and global `fetch`.

Recommended process shape:

```bash
cd /data/yangr/qwen3-tts-poc/node-realtime-demo
nohup /data/yangr/runtime/node-v20/bin/node server.mjs > logs/node-demo.log 2>&1 &
echo $! > node-demo.pid
```

Health:

```bash
curl -sS http://127.0.0.1:5177/api/config
tail -100 /data/yangr/qwen3-tts-poc/node-realtime-demo/logs/node-demo.log
```

Do not commit `.env`; it contains internal endpoints and API keys.

