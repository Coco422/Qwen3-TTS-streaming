import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_tts import Qwen3TTSModel


def fake_llm_sse(text: str, delay: float = 0.03):
    for char in text:
        time.sleep(delay)
        yield char


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test appendable realtime text input.")
    parser.add_argument("--mode", choices=["custom", "base"], default="custom")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--speaker", default="Vivian")
    parser.add_argument("--language", default="Auto")
    parser.add_argument("--ref-audio", default=None)
    parser.add_argument("--text", default="你好，这是一个真正按文本流追加的实时语音合成测试。文本会像大模型 SSE 一样逐字到达。")
    parser.add_argument("--delay", type=float, default=0.03)
    parser.add_argument("--output", default="realtime_text_stream.wav")
    parser.add_argument("--stable-holdback-tokens", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    stream_kwargs = dict(
        text_chunks=fake_llm_sse(args.text, args.delay),
        language=args.language,
        emit_every_frames=8,
        decode_window_frames=80,
        overlap_samples=512,
        first_chunk_emit_every=5,
        first_chunk_decode_window=48,
        first_chunk_frames=48,
        stable_holdback_tokens=args.stable_holdback_tokens,
    )

    if args.mode == "custom":
        generator = model.stream_generate_custom_voice_realtime(
            speaker=args.speaker,
            **stream_kwargs,
        )
    else:
        if args.ref_audio is None:
            raise ValueError("--ref-audio is required for --mode base")
        voice_prompt = model.create_voice_clone_prompt(
            ref_audio=args.ref_audio,
            x_vector_only_mode=True,
        )
        generator = model.stream_generate_voice_clone_realtime(
            voice_clone_prompt=voice_prompt,
            **stream_kwargs,
        )

    start = time.time()
    chunks = []
    first_chunk_time = None

    for pcm, sr in generator:
        if first_chunk_time is None:
            first_chunk_time = time.time() - start
            print(f"first audio chunk: {first_chunk_time:.3f}s")
        chunks.append(pcm)

    if chunks:
        sf.write(args.output, np.concatenate(chunks), sr)

    print(f"chunks={len(chunks)} total={time.time() - start:.3f}s output={args.output}")


if __name__ == "__main__":
    main()
