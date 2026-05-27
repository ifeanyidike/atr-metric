#!/usr/bin/env python3
"""
VAD Variance Analysis.

Compares three VAD backends on the same audio files:
  1. Silero VAD
  2. pyannote VAD
  3. WebRTC VAD

For each backend, computes the last speech end timestamp on Moshi output audio,
then computes TOR (did speech end before interrupt_end?).

Shows that VAD variance across backends is much smaller than ASR variance,
supporting the argument that VAD-TOR (ATR) is a more stable metric than ASR-TOR.

Usage:
    python eval_vad_variance.py
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import torch
import torchaudio
import webrtcvad
import struct
from pathlib import Path
from glob import glob
from typing import Optional
from scipy import stats

# ── Constants ───────────────────────────────────────────────────────────────
SR_VAD = 16_000
MERGE_GAP = 0.3
DATA_ROOT = Path("/Users/ifeanyidike/Public/research/data/v1.5")
TASKS = ["user_interruption", "user_backchannel", "talking_to_other", "background_speech"]

# ── Audio loading ────────────────────────────────────────────────────────────
def load_wav_16k(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != SR_VAD:
        wav = torchaudio.functional.resample(wav, sr, SR_VAD)
    return wav.squeeze(0)


def merge_segments(segs, gap=MERGE_GAP):
    if not segs:
        return []
    segs = sorted(segs)
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        if s - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


# ── Silero VAD ───────────────────────────────────────────────────────────────
_silero_model = None

def _get_silero():
    global _silero_model
    if _silero_model is None:
        from silero_vad import get_speech_timestamps
        model, _ = torch.hub.load(
            "snakers4/silero-vad", model="silero_vad", trust_repo=True, onnx=False
        )
        _silero_model = (model, get_speech_timestamps)
    return _silero_model


def silero_last_speech_end(wav: torch.Tensor) -> Optional[float]:
    model, get_speech_timestamps = _get_silero()
    segs = get_speech_timestamps(wav, model, sampling_rate=SR_VAD)
    segs = merge_segments([(t["start"] / SR_VAD, t["end"] / SR_VAD) for t in segs])
    return segs[-1][1] if segs else None


# ── pyannote VAD ─────────────────────────────────────────────────────────────
_pyannote_pipeline = None

def _get_pyannote():
    global _pyannote_pipeline
    if _pyannote_pipeline is None:
        import os
        from pyannote.audio import Model
        from pyannote.audio.pipelines import VoiceActivityDetection
        token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
        model = Model.from_pretrained(
            "pyannote/segmentation-3.0", revision="main", token=token
        )
        pipeline = VoiceActivityDetection(segmentation=model)
        pipeline.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})
        _pyannote_pipeline = pipeline
    return _pyannote_pipeline


def pyannote_last_speech_end(path: Path) -> Optional[float]:
    pipeline = _get_pyannote()
    output = pipeline(str(path))
    segs = [(seg.start, seg.end) for seg in output.get_timeline()]
    segs = merge_segments(segs)
    return segs[-1][1] if segs else None


# ── WebRTC VAD ───────────────────────────────────────────────────────────────
def webrtc_last_speech_end(wav: torch.Tensor, aggressiveness: int = 2) -> Optional[float]:
    vad = webrtcvad.Vad(aggressiveness)
    # WebRTC requires 16-bit PCM at 8/16/32/48 kHz, frame durations 10/20/30ms
    frame_dur_ms = 30
    frame_samples = int(SR_VAD * frame_dur_ms / 1000)  # 480 samples at 16kHz

    # Convert to 16-bit PCM bytes
    pcm = (wav.numpy() * 32768).clip(-32768, 32767).astype(np.int16)

    # Pad to frame boundary
    pad = (-len(pcm)) % frame_samples
    if pad:
        pcm = np.concatenate([pcm, np.zeros(pad, dtype=np.int16)])

    speech_segs = []
    in_speech = False
    seg_start = 0.0

    for i in range(0, len(pcm), frame_samples):
        frame = pcm[i:i + frame_samples]
        frame_bytes = struct.pack(f"{len(frame)}h", *frame)
        t = i / SR_VAD

        try:
            is_speech = vad.is_speech(frame_bytes, SR_VAD)
        except Exception:
            is_speech = False

        if is_speech and not in_speech:
            seg_start = t
            in_speech = True
        elif not is_speech and in_speech:
            speech_segs.append((seg_start, t))
            in_speech = False

    if in_speech:
        speech_segs.append((seg_start, len(pcm) / SR_VAD))

    segs = merge_segments(speech_segs)
    return segs[-1][1] if segs else None


# ── Per-sample TOR computation ───────────────────────────────────────────────
def compute_vad_tors(task_dir: Path, task: str) -> dict:
    results = {"silero": [], "webrtc": [], "pyannote": []}
    sample_dirs = sorted([d for d in task_dir.iterdir() if d.is_dir()])

    for sample_dir in sample_dirs:
        audio_path = sample_dir / "moshi_output.wav"
        metadata_path = sample_dir / "metadata.json"
        if not audio_path.exists() or not metadata_path.exists():
            continue

        with open(metadata_path) as f:
            meta = json.load(f)
        _, interrupt_end = meta["timestamps"]

        wav = load_wav_16k(audio_path)

        # Silero
        try:
            end = silero_last_speech_end(wav)
            if end is not None:
                results["silero"].append(int(end <= interrupt_end))
        except Exception as e:
            print(f"  [silero err] {sample_dir.name}: {e}")

        # WebRTC
        try:
            end = webrtc_last_speech_end(wav)
            if end is not None:
                results["webrtc"].append(int(end <= interrupt_end))
        except Exception as e:
            print(f"  [webrtc err] {sample_dir.name}: {e}")

        # pyannote
        try:
            end = pyannote_last_speech_end(audio_path)
            if end is not None:
                results["pyannote"].append(int(end <= interrupt_end))
        except Exception as e:
            print(f"  [pyannote err] {sample_dir.name}: {e}")

    summary = {}
    rates = []
    for backend, vals in results.items():
        if vals:
            rate = sum(vals) / len(vals)
            summary[backend] = {"n": len(vals), "rate": round(rate, 4)}
            rates.append(rate)
        else:
            summary[backend] = {"n": 0, "rate": None}

    if len(rates) >= 2:
        summary["max_diff"] = round(max(rates) - min(rates), 4)
        summary["variance"] = round(float(np.var(rates)), 6)
    return summary


def main():
    print("Loading Silero VAD (cached)...")
    _get_silero()
    print("Loading pyannote segmentation-3.0...")
    _get_pyannote()

    all_results = {}
    for task in TASKS:
        task_dir = DATA_ROOT / task
        if not task_dir.exists():
            continue
        print(f"\n[EVAL] {task}")
        result = compute_vad_tors(task_dir, task)
        all_results[task] = result
        for backend, stats_d in result.items():
            if isinstance(stats_d, dict):
                print(f"  {backend:12s}: rate={stats_d.get('rate')}  n={stats_d.get('n', '-')}")
            else:
                print(f"  {backend:12s}: {stats_d}")

    output_path = Path(__file__).parent / "results_vad_variance.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
