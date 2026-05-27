# ATR: Acoustic Turn-over Rate

A VAD-based metric for evaluating overlap handling in full-duplex spoken dialogue systems — no ASR required.

> *ATR: An Acoustic Turn-over Rate Metric for Evaluating Overlap Handling in Full-Duplex Spoken Dialogue Systems*
> Ifeanyi Dike, Happy Nkanta Monday

---

## The problem with TOR

[Full-Duplex-Bench](https://arxiv.org/abs/2503.04721) introduced TOR (Turn-over Rate) to measure how often a full-duplex model yields the speaking floor during overlapping speech. TOR is computed from ASR transcripts — which means the number you get depends on which ASR system you used.

We ran two standard backends on the same [Moshi](https://arxiv.org/abs/2410.00037) outputs across all four [Full-Duplex-Bench v1.5](https://arxiv.org/abs/2507.23159) scenarios:

| Scenario | MLX Whisper | AssemblyAI | Difference |
|---|---|---|---|
| user_interruption | 0.000 | 0.000 | 0.000 |
| user_backchannel | 0.439 | 0.204 | **0.235** |
| talking_to_other | 0.360 | 0.140 | **0.220** |
| background_speech | 0.310 | 0.140 | **0.170** |

Up to 23.5 percentage points on the same audio. That is not noise — it is the metric changing based on which tool you picked.

## ATR

ATR replaces the ASR step with voice activity detection. Instead of asking "when did the model's last word end," it asks "when did the model's audio go silent." Same formula as TOR, no transcription needed.

Three independent VAD backends on the same audio:

| Scenario | Silero | WebRTC | pyannote | Max diff |
|---|---|---|---|---|
| user_interruption | 0.385 | 0.427 | 0.461 | **0.076** |
| user_backchannel | 0.269 | 0.326 | 0.315 | **0.057** |
| talking_to_other | 0.389 | 0.516 | 0.500 | **0.127** |
| background_speech | 0.458 | 0.605 | 0.588 | **0.147** |

2–3× more consistent than ASR-based TOR, across every scenario.

## Usage

The full pipeline runs in three steps: inference → ASR transcription → evaluation.

### Step 1 — Run Moshi inference

Runs [Moshi](https://arxiv.org/abs/2410.00037) on the benchmark audio and writes `moshi_output.wav` for each sample. Requires the `moshi` conda environment with `moshi_mlx` and `rustymimi` installed.

```bash
conda activate moshi
python inference_offline.py
```

Edit `root_dir_path` and `tasks` at the top of the script to point to your data directory.

### Step 2 — Transcribe with ASR (for TOR comparison only)

Run your preferred ASR system on each `moshi_output.wav` and save word-level timestamps as JSON. The expected format is:

```json
{"chunks": [{"timestamp": [0.0, 0.5], "text": "hello"}, ...]}
```

Save as `moshi_output_mlx.json` (MLX Whisper) or `moshi_output_assemblyai.json` (AssemblyAI) alongside each audio file.

### Step 3a — ASR sensitivity analysis

Computes TOR from MLX Whisper and AssemblyAI on Moshi outputs and compares results.

```bash
python eval_moshi_asr_sensitivity.py \
    --data_root /path/to/full-duplex-bench-v1.5 \
    --output results_moshi_asr_sensitivity.json
```

Expected data layout:
```
data_root/
  user_interruption/
    1/
      moshi_output.wav
      moshi_output_mlx.json       # MLX Whisper transcript
      moshi_output_assemblyai.json  # AssemblyAI transcript
      metadata.json
    2/ ...
  user_backchannel/ ...
  talking_to_other/ ...
  background_speech/ ...
```

### Step 3b — VAD variance analysis

Computes ATR from three VAD backends and reports pairwise disagreement.

```bash
python eval_vad_variance.py
```

Edit `DATA_ROOT` at the top of the script to point to your data directory.

### Requirements

```bash
pip install silero-vad webrtcvad pyannote.audio torchaudio soundfile
```

For pyannote, you need a HuggingFace account with access to
[pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).
Log in with:

```bash
huggingface-cli login
```

## Repository structure

```
atr_paper.tex                  # Paper (LaTeX)
atr_refs.bib                   # Bibliography
inference_offline.py           # Step 1: Moshi offline inference
eval_moshi_asr_sensitivity.py  # Step 3a: ASR sensitivity analysis (TOR)
eval_vad_variance.py           # Step 3b: VAD variance analysis (ATR)
```

## Citation

```bibtex
@article{dike2026atr,
  title  = {ATR: An Acoustic Turn-over Rate Metric for Evaluating Overlap Handling in Full-Duplex Spoken Dialogue Systems},
  author = {Ifeanyi Dike and Happy Nkanta Monday},
  year   = {2026}
}
```

## License

MIT
