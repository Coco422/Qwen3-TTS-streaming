# Qwen3-TTS Realtime Architecture

This document describes how the realtime path in this fork works, why it is different from sentence-based chunking, and which parts of the code implement it.

## Summary

The realtime path keeps one TTS generation session open while upstream text is still arriving. LLM SSE deltas are appended into a thread-safe text buffer, and the TTS generation loop reads from that buffer, waits for more tokenizer-visible text when needed, and emits PCM audio chunks as soon as enough acoustic frames are available.

This is not punctuation-based sentence splitting. The client does not wait for a period or a full sentence before calling TTS. It keeps feeding the same TTS session:

```text
LLM SSE delta
  -> input_text_buffer.append(delta)
  -> realtime hidden-state provider extends text context
  -> talker/code predictor generates speech codec frames
  -> decoder emits small PCM chunks
  -> WebSocket response.audio.delta
```

The practical benefit is continuity. Audio chunks are generated from one ongoing model state instead of many independent TTS requests, so prosody, rhythm, and speaker state are less likely to reset at every sentence boundary.

## Main Code Paths

| Area | File | Role |
| --- | --- | --- |
| Appendable text source | `qwen_tts/inference/qwen3_tts_model.py` | `RealtimeTextInputBuffer` provides `append()`, `finish()`, and abort semantics for cross-thread text streaming. |
| Realtime text prefill | `qwen_tts/inference/qwen3_tts_model.py` | `_prepare_realtime_text_inputs()` reads just enough initial text to create the first model inputs. |
| Hidden-state extension | `qwen_tts/inference/qwen3_tts_model.py` | `_RealtimeTextHiddenProvider` keeps consuming later text chunks and exposes hidden states to the generation loop. |
| CustomVoice realtime API | `qwen_tts/inference/qwen3_tts_model.py` | `stream_generate_custom_voice_realtime()` accepts `text_chunks` instead of a completed text string. |
| Base clone realtime API | `qwen_tts/inference/qwen3_tts_model.py` | `stream_generate_voice_clone_realtime()` supports appendable text with `x_vector_only_mode=True`. |
| Ali-style WebSocket | `examples/realtime_ws_server.py` | Accepts `session.update`, `input_text_buffer.append`, `input_text_buffer.commit`, and `session.finish`; returns `response.audio.delta`. |
| LLM SSE proxy | `examples/realtime_ws_server.py` | Optional `llm.sse.start` path streams OpenAI-compatible SSE deltas directly into the text buffer. |

## End-to-End Flow

```text
Browser / client
  -> WebSocket connect /api-ws/v1/realtime?model=qwen-tts-realtime
  -> session.update
  -> input_text_buffer.append for each LLM delta
  -> session.finish when LLM is done

examples/realtime_ws_server.py
  -> RealtimeTextInputBuffer.append(delta)
  -> stream_generate_custom_voice_realtime(text_chunks=buffer)
  -> response.audio.delta with base64 pcm_f32le

Client playback
  -> decode base64 PCM
  -> enqueue to WebAudio / audio sink
```

## Why It Can Start Before a Full Sentence

The realtime path separates three concerns:

1. Initial text readiness: the server can wait for only a few tokenizer-visible characters before starting generation. The WebSocket demo exposes this as `--initial-buffer-chars`.
2. Stable tokenizer tail: `stable_holdback_tokens` keeps a small BPE tail out of the committed hidden-state stream so character-by-character input does not expose unstable merge prefixes.
3. Audio emission cadence: generation emits PCM after a small number of codec frames, controlled by `first_chunk_emit_every`, `first_chunk_frames`, and `emit_every_frames`.

The first chunk can therefore use aggressive settings:

```bash
--first-chunk-emit-every 1
--first-chunk-decode-window 8
--first-chunk-frames 2
```

Then the stream switches to more stable chunking:

```bash
--emit-every-frames 16
--decode-window-frames 64
--overlap-samples 256
```

## Audio Chunking and Continuity

Qwen3-TTS generates speech codec/acoustic frames before decoding to PCM. The realtime implementation emits audio once enough new frames are available. Each emitted block is decoded with a context window and can be overlapped with the previous block. This avoids the most obvious discontinuity from independent TTS calls.

The important distinction:

```text
Sentence chunking:
  text sentence A -> TTS request A -> audio A
  text sentence B -> TTS request B -> audio B

Realtime:
  one session
  text deltas A+B+C... -> one ongoing TTS generation state -> audio chunks
```

## Concurrency Model

The WebSocket server intentionally uses a per-process generation lock:

```python
self.generation_lock = threading.Lock()
```

and a single TTS worker:

```python
self.tts_executor = ThreadPoolExecutor(max_workers=1)
```

This was introduced with this fork's realtime WebSocket path; it was not present in the upstream repository before `examples/realtime_ws_server.py` was added. The lock prevents two realtime sessions from mutating or consuming one model instance at the same time.

Current production recommendation:

```text
1 process / 1 loaded model instance / 1 realtime generation at a time
N concurrent realtime users -> N service instances behind a small router/load balancer
```

A true multi-session engine would require deeper batching or paged-state work inside the model generation loop. That is a separate implementation, not just a server flag.

## LLM Thinking Control

For OpenAI-compatible Qwen reasoning models served by vLLM, this fork sends:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

by default in the LLM SSE proxy. This keeps first-token latency lower and prevents hidden reasoning text from being spoken. Use `--llm-enable-thinking` only when reasoning text is intentionally desired.

## Flash-Attn Notes

`examples/realtime_ws_server.py` loads the model with:

```python
attn_implementation=None if args.no_flash_attn else "flash_attention_2"
```

If `flash-attn` is not installed, use `--no-flash-attn`. On the tested RTX 4090 environment, the compatible wheel was:

```text
flash_attn-2.8.3.post1+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```

The tested 0.6B realtime path did not show a stable large improvement from flash-attn. The main latency wins came from keeping the session appendable, using fast codebook generation, CUDA graph capture for the codebook loop, and aggressive first-chunk settings.

