#!/usr/bin/env python3
"""Convert a YouTube video's speech to a readable Word document.

Usage:
  python youtube_to_word.py --url <youtube_url> --output output.docx

Dependencies:
  pip install yt-dlp openai-whisper python-docx

Notes:
  - Requires ffmpeg installed and available on PATH.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
from pathlib import Path

import whisper
from docx import Document
from yt_dlp import YoutubeDL


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def download_audio(url: str, output_dir: Path) -> Path:
    output_template = str(output_dir / "audio.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title") or "YouTube Transcript"
    audio_path = output_dir / "audio.mp3"
    if not audio_path.exists():
        raise FileNotFoundError("Failed to download audio from the YouTube URL.")
    return audio_path, title


def transcribe_audio(audio_path: Path, model_name: str) -> str:
    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path))
    return result["text"].strip()


def chunk_sentences(text: str, sentences_per_paragraph: int = 3) -> list[str]:
    sentences = SENTENCE_RE.split(text)
    paragraphs = []
    buffer: list[str] = []
    for sentence in sentences:
        if not sentence:
            continue
        buffer.append(sentence.strip())
        if len(buffer) >= sentences_per_paragraph:
            paragraphs.append(" ".join(buffer))
            buffer = []
    if buffer:
        paragraphs.append(" ".join(buffer))
    return paragraphs


def write_docx(title: str, paragraphs: list[str], output_path: Path) -> None:
    document = Document()
    document.add_heading(title, level=1)
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    document.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YouTube speech-to-Word converter")
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument(
        "--output",
        required=True,
        help="Output .docx path",
    )
    parser.add_argument(
        "--model",
        default="base",
        help="Whisper model name (tiny, base, small, medium, large)",
    )
    parser.add_argument(
        "--sentences",
        type=int,
        default=3,
        help="Sentences per paragraph",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    if output_path.suffix.lower() != ".docx":
        raise ValueError("Output path must end with .docx")

    if not shutil.which("ffmpeg"):
        raise EnvironmentError("ffmpeg is required and must be on PATH.")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        audio_path, title = download_audio(args.url, temp_path)
        transcript = transcribe_audio(audio_path, args.model)

    paragraphs = chunk_sentences(transcript, args.sentences)
    write_docx(title, paragraphs, output_path)
    print(f"Saved Word document to: {output_path}")


if __name__ == "__main__":
    main()
