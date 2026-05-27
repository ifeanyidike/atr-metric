#!/usr/bin/env python3
"""
Offline batch inference for Moshi on Full-Duplex-Bench v1.5.
Uses rustymimi + moshi_mlx directly — no WebSocket server needed.

Usage:
    conda activate moshi
    python inference_offline.py
"""

from __future__ import annotations

import time
from glob import glob
from pathlib import Path

import huggingface_hub
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import rustymimi
import sentencepiece
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F_torch
from moshi_mlx import models, utils

### Configuration ###
HF_REPO = "kyutai/moshiko-mlx-bf16"
root_dir_path = Path("/Users/ifeanyidike/Public/research/data/v1.5")
tasks = [
    "user_interruption",
    "user_backchannel",
    "talking_to_other",
    "background_speech",
]
prefix = ""          # "" or "clean_"
overwrite = True
OUTPUT_NAME = "moshi_output.wav"
#####################

SEND_SR = 24_000
FRAME_SMP = 1_920


def load_weights():
    print("[INFO] Downloading model weights...")
    model_file = huggingface_hub.hf_hub_download(HF_REPO, "model.safetensors")
    mimi_file = huggingface_hub.hf_hub_download(
        HF_REPO, "tokenizer-e351c8d8-checkpoint125.safetensors"
    )

    print("[INFO] Loading LM...")
    lm_config = models.config_v0_1()
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    model.load_weights(model_file, strict=True)

    print("[INFO] Warming up...")
    model.warmup(None)

    print("[INFO] Model ready.")
    return model, mimi_file


def make_gen_and_mimi(model, mimi_file, max_steps):
    """Create fresh gen and mimi instances for each file."""
    gen = models.LmGen(
        model=model,
        max_steps=max_steps,
        text_sampler=utils.Sampler(),
        audio_sampler=utils.Sampler(),
        check=False,
    )
    mimi = rustymimi.StreamTokenizer(mimi_file, num_codebooks=8)
    return gen, mimi


def load_wav_24k(path: Path) -> np.ndarray:
    """Load wav, convert to mono float32 at 24kHz."""
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SEND_SR:
        t = torch.from_numpy(wav).unsqueeze(0)
        t = F_torch.resample(t, sr, SEND_SR)
        wav = t.squeeze(0).numpy().astype(np.float32)
    return wav.astype(np.float32)


def pad_to_frame(wav: np.ndarray) -> np.ndarray:
    pad = (-len(wav)) % FRAME_SMP
    if pad:
        wav = np.concatenate([wav, np.zeros(pad, dtype=np.float32)])
    return wav


def run_inference(model, mimi_file, inp_path: Path, out_path: Path):
    wav = pad_to_frame(load_wav_24k(inp_path))
    n_frames = len(wav) // FRAME_SMP
    gen, mimi = make_gen_and_mimi(model, mimi_file, max_steps=n_frames + 50)
    out_pcm = []

    for i in range(n_frames):
        chunk = wav[i * FRAME_SMP : (i + 1) * FRAME_SMP]

        # Encode input audio chunk → tokens
        mimi.encode(chunk)
        in_tokens = mimi.get_encoded()
        if in_tokens is None:
            in_tokens = np.zeros((8, 1), dtype=np.uint32)

        # Step the LM
        in_mx = mx.array(in_tokens).transpose(1, 0)[:, : gen.main_codebooks]
        gen.step(in_mx, ct=None)

        # Get output audio tokens → decode to PCM
        audio_tokens = gen.last_audio_tokens()
        if audio_tokens is not None:
            codes = np.array(audio_tokens).astype(np.uint32)
            mimi.decode(codes)
            pcm = mimi.get_decoded()
            if pcm is not None and len(pcm) > 0:
                out_pcm.append(pcm)

    # Flush remaining decoded frames
    for _ in range(8):
        pcm = mimi.get_decoded()
        if pcm is not None and len(pcm) > 0:
            out_pcm.append(pcm)

    if out_pcm:
        audio = np.concatenate(out_pcm).astype(np.float32)
    else:
        audio = np.zeros(len(wav), dtype=np.float32)

    # Trim or pad to match input length
    target_len = len(wav)
    if len(audio) > target_len:
        audio = audio[:target_len]
    elif len(audio) < target_len:
        audio = np.concatenate([audio, np.zeros(target_len - len(audio), dtype=np.float32)])

    sf.write(out_path, audio, SEND_SR, subtype="PCM_16")


def main():
    model, mimi_file = load_weights()

    input_files = []
    for t in tasks:
        pattern = str(root_dir_path / t / "*" / f"{prefix}input.wav")
        input_files += sorted(glob(pattern))

    print(f"[INFO] {len(input_files)} files to process.")

    for inp_str in input_files:
        inp = Path(inp_str)
        out = inp.with_name(inp.name.replace("input.wav", OUTPUT_NAME))

        if not overwrite and out.exists():
            print(f"[SKIP] {inp}")
            continue

        print(f"[RUN] {inp}")
        t0 = time.time()
        try:
            run_inference(model, mimi_file, inp, out)
            print(f"[DONE] {out.name} ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"[ERR] {inp}: {e}")


if __name__ == "__main__":
    main()
