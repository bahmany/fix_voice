"""Persian TTS via F5-TTS Persian fine-tune (CPU)."""
import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from f5_tts.api import F5TTS

REPO = Path(__file__).parent / "f5-persian"
SAMPLE_RATE = 24000
SENT_PAUSE = 0.18
CROSSFADE = 0.03


def load_refs():
    data = json.load(open("/tmp/best_speeches.json"))
    return {e["reference_audio"].split("/")[-1]: e["transcript"] for e in data}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--ref", default=str(REPO / "refs" / "ref_11_twenties_female_speaker9713_040.wav"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--nfe", type=int, default=16)
    args = ap.parse_args()

    refs = load_refs()
    ref_name = Path(args.ref).stem + ".wav"
    ref_text = refs.get(ref_name)
    if not ref_text:
        sys.exit(f"no transcript for {ref_name}")

    print("loading F5-TTS (CPU)...", flush=True)
    f5 = F5TTS(
        model="F5TTS_v1_Base",
        ckpt_file=str(REPO / "model_last.pt"),
        vocab_file=str(REPO / "vocab.txt"),
        use_ema=False,
        device="cpu",
    )

    text = open(args.text_file, encoding="utf-8").read().strip()
    # Full sentences only — F5 handles intra-sentence prosody from punctuation.
    sentences = [s.strip() for s in re.split(r"(?<=[.!؟])\s+", text) if s.strip()]

    pieces = []
    for si, sent in enumerate(sentences, 1):
        print(f"  [{si}/{len(sentences)}] {sent[:50]}...", flush=True)
        wav, sr, _ = f5.infer(
            ref_file=args.ref, ref_text=ref_text, gen_text=sent,
            nfe_step=args.nfe, speed=1.0, seed=None,
        )
        wav = np.asarray(wav, dtype=np.float32)
        if sr != SAMPLE_RATE:
            import soxr
            wav = soxr.resample(wav, sr, SAMPLE_RATE)
        pieces.append(wav)
        pieces.append(np.zeros(int(SENT_PAUSE * SAMPLE_RATE), dtype=np.float32))

    full = np.concatenate(pieces)
    sf.write(args.out, full, SAMPLE_RATE)
    print(f"written {args.out} ({len(full) / SAMPLE_RATE:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
