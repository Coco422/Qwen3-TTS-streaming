import argparse
import asyncio
import base64
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


app = FastAPI()
SERVER = None


def _torch_dtype(name: str):
    name = (name or "bfloat16").lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _pcm_f32le_b64(pcm: np.ndarray) -> str:
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    return base64.b64encode(np.ascontiguousarray(pcm).tobytes()).decode("ascii")


def _extract_openai_delta(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    first = choices[0] or {}
    delta = first.get("delta") or {}
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            return content
    message = first.get("message") or {}
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    return ""


def _iter_openai_sse(
    base_url: str,
    api_key: Optional[str],
    model: str,
    messages: Iterable[Dict[str, str]],
    temperature: Optional[float] = None,
):
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body: Dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "stream": True,
    }
    if temperature is not None:
        body["temperature"] = temperature
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = _extract_openai_delta(payload)
            if delta:
                yield delta


class RealtimeServer:
    def __init__(self, args: argparse.Namespace):
        from qwen_tts import Qwen3TTSModel, RealtimeTextInputBuffer

        self.args = args
        self.model = Qwen3TTSModel.from_pretrained(
            args.model,
            device_map=args.device,
            dtype=_torch_dtype(args.dtype),
            attn_implementation=None if args.no_flash_attn else "flash_attention_2",
        )
        self.text_buffer_cls = RealtimeTextInputBuffer
        self.generation_lock = threading.Lock()
        self.demo_html = Path(__file__).with_name("realtime_web_demo.html")

    def read_demo_html(self) -> str:
        return self.demo_html.read_text(encoding="utf-8")

    async def handle_ws(self, websocket: WebSocket) -> None:
        await websocket.accept()
        loop = asyncio.get_running_loop()
        out_queue: asyncio.Queue = asyncio.Queue()
        text_buffer = self.text_buffer_cls()
        threads = []
        state: Dict[str, Any] = {
            "model_mode": self.args.mode,
            "commit_mode": "server_commit",
            "speaker": self.args.speaker,
            "language": self.args.language,
            "started": False,
            "closed": False,
            "stable_holdback_tokens": self.args.stable_holdback_tokens,
        }

        def enqueue(event: Dict[str, Any]) -> None:
            event.setdefault("created_at", time.time())
            loop.call_soon_threadsafe(out_queue.put_nowait, event)

        async def sender() -> None:
            while True:
                event = await out_queue.get()
                await websocket.send_text(_json_dumps(event))

        def run_tts() -> None:
            if not self.generation_lock.acquire(blocking=False):
                text_buffer.abort(RuntimeError("Another realtime TTS session is already running"))
                enqueue({
                    "type": "error",
                    "code": "busy",
                    "message": "Another realtime TTS session is already running",
                })
                return
            started_at = time.time()
            first_audio_sent = False
            try:
                enqueue({"type": "response.created"})
                stream_kwargs = dict(
                    text_chunks=text_buffer,
                    language=state["language"],
                    emit_every_frames=self.args.emit_every_frames,
                    decode_window_frames=self.args.decode_window_frames,
                    overlap_samples=self.args.overlap_samples,
                    first_chunk_emit_every=self.args.first_chunk_emit_every,
                    first_chunk_decode_window=self.args.first_chunk_decode_window,
                    first_chunk_frames=self.args.first_chunk_frames,
                    stable_holdback_tokens=state["stable_holdback_tokens"],
                )
                if state["model_mode"] == "custom":
                    generator = self.model.stream_generate_custom_voice_realtime(
                        speaker=state["speaker"],
                        **stream_kwargs,
                    )
                else:
                    if not self.args.ref_audio:
                        raise ValueError("--ref-audio is required when --mode=base")
                    prompt = self.model.create_voice_clone_prompt(
                        ref_audio=self.args.ref_audio,
                        x_vector_only_mode=True,
                    )
                    generator = self.model.stream_generate_voice_clone_realtime(
                        voice_clone_prompt=prompt,
                        **stream_kwargs,
                    )
                for pcm, sample_rate in generator:
                    event = {
                        "type": "response.audio.delta",
                        "delta": _pcm_f32le_b64(pcm),
                        "sample_rate": sample_rate,
                        "format": "pcm_f32le",
                    }
                    if not first_audio_sent:
                        first_audio_sent = True
                        event["first_audio_ms"] = round((time.time() - started_at) * 1000, 2)
                    enqueue(event)
                enqueue({"type": "response.audio.done"})
                enqueue({"type": "response.done"})
                enqueue({"type": "session.finished"})
            except Exception as exc:
                enqueue({
                    "type": "error",
                    "code": type(exc).__name__,
                    "message": str(exc),
                })
            finally:
                self.generation_lock.release()

        def start_tts_once() -> None:
            if state["started"]:
                return
            state["started"] = True
            thread = threading.Thread(target=run_tts, daemon=True)
            threads.append(thread)
            thread.start()

        def run_llm_sse(payload: Dict[str, Any]) -> None:
            if not self.args.llm_base_url or not self.args.llm_model:
                text_buffer.abort(RuntimeError("LLM SSE proxy is not configured on the server"))
                enqueue({
                    "type": "error",
                    "code": "llm_not_configured",
                    "message": "Start server with --llm-base-url and --llm-model to use llm.sse.start",
                })
                return
            prompt = payload.get("prompt")
            messages = payload.get("messages")
            if not messages:
                messages = [{"role": "user", "content": prompt or ""}]
            try:
                for delta in _iter_openai_sse(
                    base_url=self.args.llm_base_url,
                    api_key=self.args.llm_api_key,
                    model=self.args.llm_model,
                    messages=messages,
                    temperature=payload.get("temperature"),
                ):
                    enqueue({"type": "response.text.delta", "delta": delta})
                    text_buffer.append(delta)
                enqueue({"type": "response.text.done"})
                text_buffer.finish()
            except (urllib.error.URLError, TimeoutError, Exception) as exc:
                text_buffer.abort(exc)
                enqueue({
                    "type": "error",
                    "code": type(exc).__name__,
                    "message": str(exc),
                })

        sender_task = asyncio.create_task(sender())
        await websocket.send_text(_json_dumps({
            "type": "session.created",
            "session": {
                "model_mode": state["model_mode"],
                "mode": state["commit_mode"],
                "speaker": state["speaker"],
                "language": state["language"],
                "response_format": "pcm_f32le",
            },
        }))
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_text(_json_dumps({
                        "type": "error",
                        "code": "bad_json",
                        "message": "Expected a JSON event",
                    }))
                    continue

                event_type = event.get("type")
                if event_type == "session.update":
                    session = event.get("session") or {}
                    if session.get("model_mode") in {"custom", "base"}:
                        state["model_mode"] = session["model_mode"]
                    if session.get("mode"):
                        state["commit_mode"] = session["mode"]
                    state["speaker"] = session.get("speaker", state["speaker"])
                    state["language"] = session.get("language", state["language"])
                    state["stable_holdback_tokens"] = int(
                        session.get("stable_holdback_tokens", state["stable_holdback_tokens"])
                    )
                    enqueue({"type": "session.updated", "session": dict(state)})
                elif event_type == "input_text_buffer.append":
                    text = event.get("text") or event.get("delta") or ""
                    text_buffer.append(text)
                    start_tts_once()
                    enqueue({"type": "input_text_buffer.appended"})
                elif event_type == "input_text_buffer.commit":
                    start_tts_once()
                    enqueue({"type": "input_text_buffer.committed"})
                elif event_type == "session.finish":
                    text_buffer.finish()
                    enqueue({"type": "session.finish.received"})
                elif event_type == "llm.sse.start":
                    start_tts_once()
                    thread = threading.Thread(target=run_llm_sse, args=(event,), daemon=True)
                    threads.append(thread)
                    thread.start()
                    enqueue({"type": "llm.sse.started"})
                else:
                    enqueue({
                        "type": "error",
                        "code": "unknown_event",
                        "message": f"Unknown event type: {event_type}",
                    })
        except WebSocketDisconnect:
            pass
        finally:
            state["closed"] = True
            text_buffer.abort(RuntimeError("WebSocket disconnected"))
            sender_task.cancel()


@app.get("/")
async def index():
    if SERVER is None:
        return HTMLResponse("Server is starting", status_code=503)
    return HTMLResponse(SERVER.read_demo_html())


@app.get("/health")
async def health():
    return JSONResponse({"ok": True})


@app.websocket("/v1/realtime/tts")
async def realtime_tts(websocket: WebSocket):
    if SERVER is None:
        await websocket.close(code=1011)
        return
    await SERVER.handle_ws(websocket)


def parse_args():
    parser = argparse.ArgumentParser(description="Ali-style realtime Qwen3-TTS WebSocket demo server.")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    parser.add_argument("--mode", choices=["custom", "base"], default="custom")
    parser.add_argument("--ref-audio", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--no-flash-attn", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--speaker", default="Vivian")
    parser.add_argument("--language", default="Auto")
    parser.add_argument("--emit-every-frames", type=int, default=8)
    parser.add_argument("--decode-window-frames", type=int, default=80)
    parser.add_argument("--overlap-samples", type=int, default=512)
    parser.add_argument("--first-chunk-emit-every", type=int, default=5)
    parser.add_argument("--first-chunk-decode-window", type=int, default=48)
    parser.add_argument("--first-chunk-frames", type=int, default=48)
    parser.add_argument("--stable-holdback-tokens", type=int, default=1)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-model", default=None)
    return parser.parse_args()


def main():
    global SERVER
    args = parse_args()
    SERVER = RealtimeServer(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
