import time

import numpy as np
import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel


def fake_llm_sse(text: str, delay: float = 0.03):
    for char in text:
        time.sleep(delay)
        yield char


model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

voice_prompt = model.create_voice_clone_prompt(
    ref_audio="kuklina-1.wav",
    x_vector_only_mode=True,
)

text = "你好，这是一个真正按文本流追加的实时语音合成测试。文本会像大模型 SSE 一样逐字到达。"

start = time.time()
chunks = []
first_chunk_time = None

for pcm, sr in model.stream_generate_voice_clone_realtime(
    text_chunks=fake_llm_sse(text),
    language="Auto",
    voice_clone_prompt=voice_prompt,
    emit_every_frames=8,
    decode_window_frames=80,
    overlap_samples=512,
    first_chunk_emit_every=5,
    first_chunk_decode_window=48,
    first_chunk_frames=48,
):
    if first_chunk_time is None:
        first_chunk_time = time.time() - start
        print(f"first audio chunk: {first_chunk_time:.3f}s")
    chunks.append(pcm)

if chunks:
    sf.write("realtime_text_stream.wav", np.concatenate(chunks), sr)

print(f"chunks={len(chunks)} total={time.time() - start:.3f}s")
