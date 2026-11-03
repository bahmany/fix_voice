"""Render N candidates per voice, ASR each, keep the cleanest vs source text."""
import argparse
import difflib
import json
import re
import subprocess
import urllib.request

GW = "http://127.0.0.1:8080"
KEY = "sk_IExlQdp6VPK6kGoKw1fQsQWVmCgfrBNwJCdJFNzD"


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{GW}{path}", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def transcribe(path: str) -> str:
    out = subprocess.run(
        ["curl", "-s", "-m", "120", "-X", "POST", f"{GW}/v1/audio/transcriptions",
         "-H", f"Authorization: Bearer {KEY}",
         "-F", f"file=@{path}", "-F", "model=shenava"],
        capture_output=True, text=True)
    try:
        return json.loads(out.stdout).get("text", "")
    except (json.JSONDecodeError, TypeError):
        return ""


def norm(t: str) -> str:
    t = re.sub(r"[\u200c ]", "", t)
    return re.sub(r"[،؛؟!.\n]", "", t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--voices", nargs="+", required=True)
    ap.add_argument("--model", default="pockettts-farsi-v2")
    ap.add_argument("--candidates", type=int, default=3)
    ap.add_argument("--prefix", default="hq")
    args = ap.parse_args()

    text = open(args.text_file, encoding="utf-8").read().strip()
    ref = norm(text)
    for voice in args.voices:
        best, best_score, best_file = None, -1.0, None
        for i in range(1, args.candidates + 1):
            f = f"{args.prefix}_{voice}_c{i}.wav"
            subprocess.run(["curl", "-s", "-m", "300", "-X", "POST", f"{GW}/v1/audio/speech",
                            "-H", f"Authorization: Bearer {KEY}",
                            "-H", "Content-Type: application/json",
                            "-d", json.dumps({"model": args.model, "input": text, "voice": voice}),
                            "-o", f], check=True)
            hyp = norm(transcribe(f))
            score = difflib.SequenceMatcher(None, ref, hyp).ratio()
            print(f"  {f}: score {score:.3f}", flush=True)
            if score > best_score:
                best, best_score, best_file = hyp, score, f
        import shutil
        shutil.copy(best_file, f"{args.prefix}_{voice}.wav")
        print(f"{voice}: kept {best_file} (score {best_score:.3f})", flush=True)


if __name__ == "__main__":
    main()
