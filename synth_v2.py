"""Persian TTS via pocket-tts-farsi-v2 (phoneme input, sentence-chunked)."""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parent / "pocket-tts-farsi-v2"))
from normalize_fa import normalize_for_model  # noqa: E402

from pocket_tts import TTSModel  # noqa: E402

REPO = Path(__file__).parent / "pocket-tts-farsi-v2"
G2P = "mehdi-hf/Homo-GE2PE-Persian-HF"
TO_PHONEMES = str.maketrans({"/": "a", "a": "A", "@": "?", "$": "S", "c": "C"})
SAMPLE_RATE = 24000
PAUSE_BY_MARK = {"…": 0.35, ".": 0.20, "؟": 0.25, "!": 0.25, "؛": 0.28}
CHUNK_PAUSE = 0.06
CHUNK_BUDGET = 18
MAX_RETRIES = 4
FADE = 0.015  # linear fade in/out on every piece to kill hard seams


def _fade(x: np.ndarray) -> np.ndarray:
    n = int(FADE * SAMPLE_RATE)
    if len(x) > 2 * n:
        x[:n] *= np.linspace(0, 1, n)
        x[-n:] *= np.linspace(1, 0, n)
    return x


def load_g2p():
    tok = AutoTokenizer.from_pretrained(G2P)
    g2p = T5ForConditionalGeneration.from_pretrained(G2P).eval()
    return tok, g2p


def phonemise(text: str, tok, g2p) -> str:
    text = normalize_for_model(text)
    text = text.replace("؟", "").replace("?", "")
    enc = tok([text], add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        out = g2p.generate(**enc, num_beams=5, max_length=512, early_stopping=True)
    raw = tok.batch_decode(out, skip_special_tokens=True)[0].strip()
    return raw.translate(TO_PHONEMES)


def split_sentences(text: str) -> list[tuple[str, str]]:
    # Returns (sentence, pause_mark) pairs; splits on . ! ؟ … ؛ and newlines.
    parts = re.split(r"([.…؟!؛])|\n+", text)
    out, buf, mark = [], "", "."
    for p in parts:
        if p is None:
            continue
        if p in PAUSE_BY_MARK:
            if buf.strip():
                out.append((buf.strip(), p))
            buf, mark = "", "."
        else:
            buf += " " + p
    if buf.strip():
        out.append((buf.strip(), "."))
    return out


def chunk_phonemes(phon: str, spm) -> list[str]:
    # Keep the ezafe marker "1" glued to the preceding word.
    words = re.findall(r"\S+1?|1", phon) if "1" in phon else phon.split()
    words, merged = phon.split(), []
    i = 0
    while i < len(words):
        if words[i] == "1" and merged:
            merged[-1] += "1"
            i += 1
        else:
            merged.append(words[i])
            i += 1
    words = merged

    chunks, cur, cur_len = [], [], 0
    for w in words:
        w_len = len(spm.encode(w))
        if cur and cur_len + w_len > CHUNK_BUDGET and cur_len >= 9:
            chunks.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(w)
        cur_len += w_len
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def synth_chunk(model, state, phon: str, spm) -> np.ndarray:
    n_tok = len(spm.encode(phon))
    cap = n_tok / 3.0 + 2.0
    for attempt in range(MAX_RETRIES):
        audio = model.generate_audio(state, phon, frames_after_eos=0)
        dur = len(audio) / SAMPLE_RATE
        if dur <= cap:
            return np.asarray(audio, dtype=np.float32)
        print(f"    runaway ({dur:.1f}s > cap {cap:.1f}s), retry {attempt + 1}", flush=True)
    return np.asarray(audio, dtype=np.float32)  # best effort after retries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--voice", default=str(REPO / "samples" / "prompt_hello.wav"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    print("loading G2P...", flush=True)
    tok, g2p = load_g2p()
    print("loading TTS model...", flush=True)
    model = TTSModel.load_model(config=str(REPO / "model.yaml"), temp=0.3)
    state = model.get_state_for_audio_prompt(args.voice)
    spm_path = REPO / "tokenizer_ph.model"
    import sentencepiece as spm_mod
    spm = spm_mod.SentencePieceProcessor(model_file=str(spm_path))

    sentences = split_sentences(args.text)
    pieces = []
    for si, (sent, mark) in enumerate(sentences, 1):
        phon = phonemise(sent, tok, g2p)
        print(f"  sentence {si} [{mark}]: {phon[:70]}...", flush=True)
        chunks = chunk_phonemes(phon, spm)
        for ci, ch in enumerate(chunks):
            ch = ch.replace("1", "")  # ezafe marker is for chunking only
            print(f"    chunk {ci + 1}/{len(chunks)} ({len(spm.encode(ch))} tok)", flush=True)
            audio = _fade(synth_chunk(model, state, ch, spm))
            pieces.append(audio)
            pieces.append(np.zeros(int(CHUNK_PAUSE * SAMPLE_RATE), dtype=np.float32))
        pieces[-1] = np.zeros(int(PAUSE_BY_MARK.get(mark, 0.20) * SAMPLE_RATE), dtype=np.float32)

    full = np.concatenate(pieces)
    sf.write(args.out, full, SAMPLE_RATE)
    print(f"written {args.out} ({len(full) / SAMPLE_RATE:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
