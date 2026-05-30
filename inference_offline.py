"""
GPT-4o Realtime offline inference for Full-Duplex-Bench v1.5.

Sends each input.wav to the GPT-4o Realtime API over WebSocket and
saves the audio response as gpt_output.wav in the same sample directory.

Usage:
    export OPENAI_API_KEY=sk-...
    python inference_offline.py

Requires:
    pip install websockets soundfile numpy torchaudio
"""

import asyncio
import base64
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import numpy as np
import soundfile as sf
import torchaudio
import websockets

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY      = os.environ.get("OPENAI_API_KEY", "")
MODEL        = "gpt-realtime"
VOICE        = "verse"
WS_URL       = f"wss://api.openai.com/v1/realtime?model={MODEL}"

DATA_ROOT    = Path(os.environ.get(
    "FDB_DATA_ROOT",
    Path(__file__).resolve().parents[3] / "data/v1.5/user_interruption"
))
OUTPUT_NAME  = "gpt_output.wav"
TARGET_SR    = 24000   # GPT-4o realtime outputs 24kHz PCM16
INPUT_SR     = 24000   # send at 24kHz PCM16
TASKS        = list(range(1, 51))   # first 50 samples only (~$2.25)
OVERWRITE    = False

# ── Audio helpers ─────────────────────────────────────────────────────────────

def load_wav_24k(path: Path) -> np.ndarray:
    """Load wav file, resample to 24kHz mono, return float32 array."""
    wav, sr = torchaudio.load(str(path))
    if sr != INPUT_SR:
        wav = torchaudio.functional.resample(wav, sr, INPUT_SR)
    wav = wav.mean(0)  # mono
    return wav.numpy().astype(np.float32)


def float32_to_pcm16_b64(audio: np.ndarray) -> str:
    """Convert float32 numpy array to base64-encoded PCM16."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    return base64.b64encode(pcm.tobytes()).decode()


def pcm16_b64_to_float32(b64: str) -> np.ndarray:
    """Convert base64 PCM16 string back to float32 numpy array."""
    raw = base64.b64decode(b64)
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    return pcm


def save_wav(path: Path, audio: np.ndarray, sr: int = TARGET_SR):
    sf.write(str(path), audio, sr, subtype="PCM_16")


# ── Realtime inference ────────────────────────────────────────────────────────

async def run_one(sample_dir: Path) -> bool:
    """Run GPT-4o Realtime on one sample. Returns True on success."""
    # Use clean_input.wav (user channel only) to avoid VAD confusion from mixed audio
    inp_path = sample_dir / "clean_input.wav"
    if not inp_path.exists():
        inp_path = sample_dir / "input.wav"
    out_path = sample_dir / OUTPUT_NAME

    if not inp_path.exists():
        print(f"  [skip] no input: {sample_dir.name}")
        return False
    if out_path.exists() and not OVERWRITE:
        print(f"  [skip] already done: {sample_dir.name}")
        return True

    audio_in = load_wav_24k(inp_path)

    headers = {"Authorization": f"Bearer {API_KEY}"}
    output_chunks = []
    success = False

    try:
        async with websockets.connect(WS_URL, additional_headers=headers) as ws:
            # 1. Wait for session.created
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if msg["type"] != "session.created":
                print(f"  [api error] {json.dumps(msg)}")
                return False

            # 2. Append user audio padded with 1.5s silence so VAD flushes all segments
            silence = np.zeros(int(INPUT_SR * 1.5), dtype=np.float32)
            padded = np.concatenate([audio_in, silence])
            audio_b64_padded = float32_to_pcm16_b64(padded)
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": audio_b64_padded,
            }))

            # 3. Collect audio delta events until response.done (completed).
            # GA model uses "response.output_audio.delta" (not "response.audio.delta").
            # If a response is cancelled (due to a second speech segment triggering
            # a new auto-response), keep waiting for the next response.done.
            deadline = time.time() + 60  # 60s max per sample
            while time.time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=10)
                except asyncio.TimeoutError:
                    break
                msg = json.loads(raw)
                t = msg.get("type", "")

                if t == "response.output_audio.delta":
                    chunk = pcm16_b64_to_float32(msg["delta"])
                    output_chunks.append(chunk)

                elif t == "response.done":
                    status = msg.get("response", {}).get("status", "")
                    if status == "completed":
                        success = True
                        break
                    # cancelled = new speech segment interrupted; keep listening
                    # for the subsequent response

                elif t == "error":
                    print(f"  [error] {msg}")
                    break

    except Exception as e:
        print(f"  [exception] {sample_dir.name}: {e}")
        return False

    if output_chunks and success:
        audio_out = np.concatenate(output_chunks)
        save_wav(out_path, audio_out)
        dur = len(audio_out) / TARGET_SR
        print(f"  [ok] {sample_dir.name}: {dur:.1f}s output")
        return True
    else:
        print(f"  [fail] {sample_dir.name}: no audio collected (success={success}, chunks={len(output_chunks)})")
        return False


async def main():
    if not API_KEY:
        raise ValueError("OPENAI_API_KEY not set")

    print(f"Running GPT-4o Realtime inference on {len(TASKS)} samples...")
    ok, fail = 0, 0

    for sid in TASKS:
        sample_dir = DATA_ROOT / str(sid)
        if not sample_dir.exists():
            print(f"  [skip] missing dir: {sid}")
            continue
        print(f"Sample {sid}...")
        result = await run_one(sample_dir)
        if result:
            ok += 1
        else:
            fail += 1
        # Small delay to avoid rate limits
        await asyncio.sleep(0.5)

    print(f"\nDone. Success: {ok}, Failed: {fail}")


if __name__ == "__main__":
    asyncio.run(main())
