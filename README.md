# ATR: Acoustic Turn-over Rate

Code and paper for **ATR (Acoustic Turn-over Rate)**, a VAD-based metric for evaluating overlap handling in full-duplex spoken dialogue systems.

> **Paper:** *ATR: An Acoustic Turn-over Rate Metric for Evaluating Overlap Handling in Full-Duplex Spoken Dialogue Systems*
> Ifeanyi Dike, Happy Nkanta Monday

## Overview

TOR (Turn-over Rate), introduced in [Full-Duplex-Bench](https://arxiv.org/abs/2503.04721), measures how often a full-duplex model yields the speaking floor during overlapping speech. It is computed from ASR transcripts — and two standard ASR backends disagree by up to 23.5 percentage points on the same audio.

ATR replaces the ASR step with voice activity detection. Three independent VAD backends (Silero, WebRTC, pyannote) agree within 7.6 pp on the same clips where ASR backends disagree by 23.5 pp.

## Repository structure

```
atr_paper.tex        # Paper source (LaTeX)
atr_refs.bib         # Bibliography
eval_vad_variance.py # VAD variance analysis (Silero vs WebRTC vs pyannote)
eval_moshi_asr_sensitivity.py  # ASR sensitivity analysis (MLX Whisper vs AssemblyAI)
```

## Requirements

```bash
pip install silero-vad webrtcvad pyannote.audio torchaudio
```

For pyannote, you need a HuggingFace token with access to `pyannote/segmentation-3.0`.

## Citation

```bibtex
@article{dike2026atr,
  title   = {ATR: An Acoustic Turn-over Rate Metric for Evaluating Overlap Handling in Full-Duplex Spoken Dialogue Systems},
  author  = {Ifeanyi Dike and Happy Nkanta Monday},
  year    = {2026}
}
```

## License

MIT
