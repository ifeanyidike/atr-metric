#!/usr/bin/env python3
"""
ASR Sensitivity Analysis for Moshi on Full-Duplex-Bench v1.5.

Computes TOR (Turn-Over Rate) from three sources:
  1. MLX Whisper ASR word boundary timestamps
  2. AssemblyAI ASR word boundary timestamps
  3. Silero VAD — audio-only, no ASR required

For each v1.5 task: user_interruption, user_backchannel, talking_to_other, background_speech.

Output: JSON summary of TOR per backend per task.

Usage:
    python eval_moshi_asr_sensitivity.py \
        --data_root /Users/ifeanyidike/Public/research/data/v1.5 \
        --output results_moshi_asr_sensitivity.json
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Optional

import torch
import torchaudio

# ── Silero VAD setup ────────────────────────────────────────────────────────
SR_VAD = 16_000
MERGE_GAP = 0.3  # merge speech segments separated by ≤300 ms

vad_module = importlib.import_module("silero_vad")
if hasattr(vad_module, "VoiceActivityDetector"):
    from silero_vad import VoiceActivityDetector
    _VAD = VoiceActivityDetector(sample_rate=SR_VAD)
    def _vad_timestamps(wav):
        return [(t["start"] / SR_VAD, t["end"] / SR_VAD) for t in _VAD.get_speech_ts(wav)]
else:
    from silero_vad import get_speech_timestamps
    _VAD_MODEL, _ = torch.hub.load(
        "snakers4/silero-vad", model="silero_vad", trust_repo=True, onnx=False
    )
    def _vad_timestamps(wav):
        return [(t["start"] / SR_VAD, t["end"] / SR_VAD)
                for t in get_speech_timestamps(wav, _VAD_MODEL, sampling_rate=SR_VAD)]


def _load_wav_mono(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != SR_VAD:
        wav = torchaudio.functional.resample(wav, sr, SR_VAD)
    return wav.squeeze(0)


def _merge_segments(segs, gap=MERGE_GAP):
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


def vad_last_speech_end(audio_path: Path) -> Optional[float]:
    """Return the end time (seconds) of the last VAD speech segment in the audio."""
    wav = _load_wav_mono(audio_path)
    segs = _merge_segments(_vad_timestamps(wav))
    if not segs:
        return None
    return segs[-1][1]


def asr_last_word_end(json_path: Path) -> Optional[float]:
    """Return the end timestamp of the last word in an ASR JSON transcript."""
    if not json_path.exists():
        return None
    with open(json_path) as f:
        data = json.load(f)
    chunks = data.get("chunks", [])
    if not chunks:
        return None
    return chunks[-1]["timestamp"][1]


def compute_tor_asr(sample_dir: Path, backend: str, interrupt_end: float) -> Optional[int]:
    """
    TOR from ASR: 1 if model's last word ended before the interruption end, else 0.
    backend: 'mlx' or 'assemblyai'
    """
    json_name = f"moshi_output_{backend}.json"
    last_end = asr_last_word_end(sample_dir / json_name)
    if last_end is None:
        return None
    return int(last_end <= interrupt_end)


def compute_tor_vad(sample_dir: Path, interrupt_end: float) -> Optional[int]:
    """
    VAD-TOR (ATR): 1 if model's last speech segment ended before the interruption end, else 0.
    """
    audio_path = sample_dir / "moshi_output.wav"
    if not audio_path.exists():
        return None
    last_end = vad_last_speech_end(audio_path)
    if last_end is None:
        return None
    return int(last_end <= interrupt_end)


def eval_task(task_dir: Path, task: str) -> dict:
    """Evaluate TOR for all backends on a single task directory."""
    results = {"mlx": [], "assemblyai": [], "vad": []}
    skipped = 0

    sample_dirs = sorted([d for d in task_dir.iterdir() if d.is_dir()])

    for sample_dir in sample_dirs:
        # Get interrupt/overlap end time from metadata
        metadata_path = sample_dir / "metadata.json"
        if not metadata_path.exists():
            skipped += 1
            continue
        with open(metadata_path) as f:
            meta = json.load(f)
        _, interrupt_end = meta["timestamps"]

        # Only user_interruption uses TOR — for other tasks we measure whether
        # model resumed (speech present after overlap end)
        if task == "user_interruption":
            for backend in ["mlx", "assemblyai"]:
                val = compute_tor_asr(sample_dir, backend, interrupt_end)
                if val is not None:
                    results[backend].append(val)
            val = compute_tor_vad(sample_dir, interrupt_end)
            if val is not None:
                results["vad"].append(val)
        else:
            # For non-interruption tasks: measure resume rate
            # Model should produce speech after the overlap window
            for backend in ["mlx", "assemblyai"]:
                json_name = f"moshi_output_{backend}.json"
                json_path = sample_dir / json_name
                if not json_path.exists():
                    continue
                with open(json_path) as f:
                    data = json.load(f)
                chunks = data.get("chunks", [])
                # Check if any word starts after overlap end
                resumed = int(any(c["timestamp"][0] > interrupt_end for c in chunks))
                results[backend].append(resumed)

            # VAD resume: any speech segment starting after overlap end
            audio_path = sample_dir / "moshi_output.wav"
            if audio_path.exists():
                wav = _load_wav_mono(audio_path)
                segs = _merge_segments(_vad_timestamps(wav))
                resumed = int(any(s > interrupt_end for s, e in segs))
                results["vad"].append(resumed)

    summary = {}
    for backend, vals in results.items():
        if vals:
            summary[backend] = {
                "n": len(vals),
                "rate": round(sum(vals) / len(vals), 4),
                "skipped": skipped,
            }
        else:
            summary[backend] = {"n": 0, "rate": None, "skipped": skipped}

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                        default="/Users/ifeanyidike/Public/research/data/v1.5")
    parser.add_argument("--output", type=str,
                        default="results_moshi_asr_sensitivity.json")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    tasks = ["user_interruption", "user_backchannel", "talking_to_other", "background_speech"]

    all_results = {}
    for task in tasks:
        task_dir = data_root / task
        if not task_dir.exists():
            print(f"[SKIP] {task} — directory not found")
            continue
        print(f"\n[EVAL] {task}")
        result = eval_task(task_dir, task)
        all_results[task] = result
        for backend, stats in result.items():
            print(f"  {backend:12s}: rate={stats['rate']}  n={stats['n']}")

    output_path = Path(__file__).parent / args.output
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
