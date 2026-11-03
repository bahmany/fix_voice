"""Prosody pass: gateway LLM re-punctuates Persian text for TTS intonation."""
import argparse
import json
import urllib.request

GATEWAY = "http://127.0.0.1:8080/v1/chat/completions"
KEY = "sk_IExlQdp6VPK6kGoKw1fQsQWVmCgfrBNwJCdJFNzD"

SYSTEM = """You are a Persian text-prosody editor for a TTS narrator engine.

The TTS engine reads plain text and uses punctuation to decide pauses and
sentence-final intonation. Your job: rewrite the passage so a narrator reads it
with natural, emotional Persian intonation.

Rules:
- Keep EVERY word exactly as written. Never add, remove, replace or reorder
  words. Only punctuation and line breaks may change.
- Use ، for short clause pauses.
- Use ؛ or : for medium pauses.
- Use … (ellipsis) for long, dramatic or wistful pauses.
- End sentences with . or ؟.
- Break long sentences into shorter ones at natural clause boundaries, so no
  sentence is longer than about 12 words.
- Put each sentence on its own line.
- Do NOT add any commentary. Output ONLY the rewritten Persian text."""

USER_TMPL = """Passage:

{text}"""


def enrich(text: str, model: str = "local-llama/qwen3.5-9b") -> str:
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TMPL.format(text=text)},
        ],
        "temperature": 0.3,
        "max_tokens": 1200,
    }).encode()
    req = urllib.request.Request(
        GATEWAY, data=body,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"].strip()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="local-llama/qwen3.5-9b")
    args = ap.parse_args()
    text = open(args.text_file, encoding="utf-8").read().strip()
    out = enrich(text, args.model)
    open(args.out, "w", encoding="utf-8").write(out + "\n")
    print(out)
