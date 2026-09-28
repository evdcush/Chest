#!/usr/bin/env python3
"""Auto-caption a directory of images for LoRA training.

Calls a local, OpenAI-compatible vision endpoint (e.g. llama.cpp's
llama-server, via your litellm proxy) to write one plain, descriptive
caption per image, optionally prefixed with a style trigger phrase.

Model choice: point this at whatever vision model you're serving. For
reliable, arbitrary-domain captioning on a single RTX 4090, Qwen3-VL-8B-
Instruct (official GGUF + mmproj from Qwen, well-supported by stock
llama.cpp) is a solid default; Qwen3-VL-32B-Instruct is a slower, higher-
quality option that still fits one 24 GB card at Q4_K_M. There is no
official "Qwen3-VL-27B" release as of this writing -- if that's what you
had in mind, double check the exact repo name before downloading.

Requires: pip install requests Pillow

Usage:
    python generate_captions.py --style 'wasuji' data/my_wasuji_images \
        --model Qwen3-VL-8B-Instruct

    # rename image+caption pairs to ref_NN.ext / ref_NN.txt afterwards
    python generate_captions.py --style 'wasuji' data/my_wasuji_images \
        --model Qwen3-VL-8B-Instruct --rename
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path

import requests
from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

PIL_FORMAT_TO_MIME = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
    "BMP": "image/bmp",
}

CAPTION_INSTRUCTION = (
    "Write one or two plain, natural-language sentences describing exactly "
    "what is visible in this image: the subject, its pose or action, the "
    "setting, and the composition. Do not mention lighting, color palette, "
    "rendering technique, art style, mood, or atmosphere. Do not speculate "
    "about anything not directly visible, such as emotions, backstory, or "
    "intent. Do not begin with \"This image shows\" or similar phrases; "
    "describe directly. Reply with only the caption itself -- no quotation "
    "marks, labels, or extra commentary."
)


class CaptionError(Exception):
    """Raised when a single image fails to caption after all retries."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-caption a directory of images for LoRA training.",
    )
    parser.add_argument(
        "directory", type=Path, help="Directory of images to caption")
    parser.add_argument(
        "--style",
        default=None,
        help='Trigger phrase to prepend, e.g. "wasuji" -> "wasuji style, ..."',
    )
    parser.add_argument(
        "--rename",
        action="store_true",
        help="After captioning, rename pairs to ref_NN.ext / ref_NN.txt",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("CAPTION_MODEL"),
        help="Model name as configured in your litellm/llama.cpp server "
        "(or set CAPTION_MODEL)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("CAPTION_API_BASE", "http://localhost:4000/v1"),
        help="OpenAI-compatible base URL (or set CAPTION_API_BASE). "
        "Default assumes a local litellm proxy on its usual port -- "
        "override if yours differs.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("CAPTION_API_KEY", ""),
        help="API key, if your server requires one (or set CAPTION_API_KEY)",
    )
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="Seconds per request")
    args = parser.parse_args(argv)
    if not args.model:
        parser.error("--model is required (or set CAPTION_MODEL)")
    return args


def find_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise CaptionError(f"Not a directory: {directory}")
    images = sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise CaptionError(f"No images found in {directory}")
    return images


def encode_image(path: Path) -> tuple[str, str]:
    """Return (base64 data, mime type), validating the file is a real image."""
    try:
        with Image.open(path) as img:
            img.verify()
            fmt = img.format
    except (UnidentifiedImageError, OSError) as exc:
        raise CaptionError(f"Not a readable image: {path.name} ({exc})") from exc
    mime = PIL_FORMAT_TO_MIME.get(fmt or "", "")
    if not mime:
        raise CaptionError(f"Unsupported image format {fmt!r}: {path.name}")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return data, mime


def clean_caption(text: str) -> str:
    text = " ".join(text.split())  # collapse whitespace/newlines
    text = text.strip().strip('"').strip("'").strip()
    return text


def request_caption(
    session: requests.Session,
    base_url: str,
    api_key: str,
    model: str,
    image_b64: str,
    mime: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> str:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CAPTION_INSTRUCTION},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                    },
                ],
            }
        ],
    }
    resp = session.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise CaptionError(f"Unexpected response shape: {data}") from exc
    caption = clean_caption(content)
    if not caption:
        raise CaptionError("Model returned an empty caption")
    return caption


def caption_one(
    session: requests.Session,
    image_path: Path,
    args: argparse.Namespace,
) -> str:
    image_b64, mime = encode_image(image_path)
    last_error: Exception | None = None
    for attempt in range(1, args.retries + 1):
        try:
            caption = request_caption(
                session,
                args.base_url,
                args.api_key,
                args.model,
                image_b64,
                mime,
                args.max_tokens,
                args.temperature,
                args.timeout,
            )
            if args.style:
                caption = f"{args.style} style, {caption}"
            return caption
        except (requests.RequestException, CaptionError) as exc:
            last_error = exc
            if attempt < args.retries:
                time.sleep(2 * attempt)
    raise CaptionError(
        f"{image_path.name}: giving up after {args.retries} attempts ({last_error})")


def write_captions(directory: Path, args: argparse.Namespace) -> list[tuple[Path, Path]]:
    images = find_images(directory)
    session = requests.Session()
    pairs: list[tuple[Path, Path]] = []
    failures: list[str] = []
    for i, image_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {image_path.name} ... ", end="", flush=True)
        try:
            caption = caption_one(session, image_path, args)
        except CaptionError as exc:
            print(f"FAILED ({exc})")
            failures.append(image_path.name)
            continue
        caption_path = image_path.with_suffix(".txt")
        caption_path.write_text(caption + "\n", encoding="utf-8")
        print("ok")
        pairs.append((image_path, caption_path))
    if failures:
        print(f"\n{len(failures)} image(s) failed and were skipped:", file=sys.stderr)
        for name in failures:
            print(f"  - {name}", file=sys.stderr)
    return pairs


def rename_pairs(pairs: list[tuple[Path, Path]]) -> None:
    if not pairs:
        return
    width = max(2, len(str(len(pairs))))
    # Two-phase rename so we never clobber a file that's already correctly named.
    temp_pairs = []
    for i, (image_path, caption_path) in enumerate(pairs, 1):
        tmp_image = image_path.with_name(f".tmp_rename_{i}{image_path.suffix}")
        tmp_caption = caption_path.with_name(f".tmp_rename_{i}.txt")
        image_path.rename(tmp_image)
        caption_path.rename(tmp_caption)
        temp_pairs.append((tmp_image, tmp_caption, image_path.suffix))
    for i, (tmp_image, tmp_caption, suffix) in enumerate(temp_pairs, 1):
        final_image = tmp_image.with_name(f"ref_{i:0{width}d}{suffix}")
        final_caption = tmp_caption.with_name(f"ref_{i:0{width}d}.txt")
        tmp_image.rename(final_image)
        tmp_caption.rename(final_caption)
        print(f"renamed -> {final_image.name}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"model:    {args.model}")
    print(f"endpoint: {args.base_url}")
    try:
        pairs = write_captions(args.directory, args)
    except CaptionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.rename:
        rename_pairs(pairs)
    print(f"\nCaptioned {len(pairs)} image(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
