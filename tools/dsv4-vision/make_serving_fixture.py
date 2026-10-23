"""Make exact prompt-plus-output profiles for the native two-rank token-ID verifier."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--context", type=int, choices=[8192, 32768, 131072, 262144], default=8192)
    parser.add_argument("--concurrency", type=int, choices=[1, 2, 4], default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--image-probe", type=Path)
    args = parser.parse_args()
    if args.max_tokens < 16 or args.max_tokens >= args.context // 2:
        parser.error("--max-tokens must be at least 16 and leave room for a saved prefix")
    if len(args.image) > 4 or (args.image and args.image_probe is None):
        parser.error("images require --image-probe and at most four --image paths")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False).ids

    prefix = encode("<｜begin▁of▁sentence｜><｜User｜>")
    expanded = len(prefix)
    image_meta = []
    marker = tokenizer.token_to_id("<｜deepseek_image｜>")
    if args.image and marker is None:
        parser.error("checkpoint tokenizer has no image placeholder")
    with tempfile.TemporaryDirectory(prefix="dsv4-profile-") as tmp:
        for i, image in enumerate(args.image):
            command = [str(args.image_probe.resolve()), str((args.checkpoint / "config.json").resolve()),
                       str(image.resolve()), str(Path(tmp) / f"patches-{i}.bf16"), str(expanded)]
            result = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
            geometry = json.loads(result.stdout)
            count = len(geometry["types"])
            image_meta.append({"at": expanded, "tokens": count,
                               "sha256": hashlib.sha256(image.read_bytes()).hexdigest()})
            prefix.append(marker)
            prefix.extend(encode("\n\n"))
            expanded += count + len(encode("\n\n"))
    pattern = encode("Analyze the following public arithmetic example. Two plus three is five. Explain each step.\n")
    if not pattern:
        parser.error("tokenizer produced an empty text pattern")
    cases = []
    for i in range(args.concurrency):
        tail = encode(f"\nCase {i}: summarize the arithmetic. <｜Assistant｜></think>")
        need = args.context - args.max_tokens - expanded - len(tail)
        if need <= 0:
            parser.error("requested images and template exceed the exact context profile")
        body = (pattern * (need // len(pattern) + 1))[:need]
        cases.append({"prompt": prefix + body + tail, "max_tokens": args.max_tokens,
                      "temperature": 0 if i % 2 == 0 else 0.8, "seed": 77 + i,
                      "images": [str(image) for image in args.image]})
    args.output.write_text(json.dumps({"cases": cases}, separators=(",", ":")), encoding="utf-8")
    meta = {"context": args.context, "concurrency": args.concurrency,
            "expanded_prompt_tokens_per_request": args.context - args.max_tokens,
            "max_output_tokens_per_request": args.max_tokens, "images": image_meta,
            "tokenizer_sha256": hashlib.sha256((args.checkpoint / "tokenizer.json").read_bytes()).hexdigest(),
            "fixture_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
