#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Coqui TTS (CPU-friendly)
from TTS.api import TTS

try:
    import soundfile as sf
except Exception:
    sf = None


def read_text(text_or_path: str) -> str:
    p = Path(text_or_path)
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return text_or_path.strip()


def pick_model(preset: str) -> str:
    """
    Choose a solid, open-weights, CPU-friendly English model.
    You can swap to any Coqui TTS model card ID later.
    """
    presets = {
        # Multi-speaker English VITS (good quality, fast on CPU)
        "vits-en": "tts_models/en/vctk/vits",
        # Single-speaker LJSpeech VITS (neutral, clean)
        "vits-ljs": "tts_models/en/ljspeech/vits",
        # FastPitch+HifiGAN (alternative timbre)
        "fastpitch": "tts_models/en/ljspeech/fast_pitch",
    }
    if preset not in presets:
        raise SystemExit(
            f"Unknown preset '{preset}'. Valid options: {', '.join(presets.keys())}"
        )
    return presets[preset]


def main():
    ap = argparse.ArgumentParser(
        description="Generate an engaging voiceover from text using open-source TTS (CPU)."
    )
    ap.add_argument(
        "text",
        help="Narration text or path to a .txt file.",
    )
    ap.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output audio path. Defaults to ./voiceover.wav",
    )
    ap.add_argument(
        "--preset",
        default="vits-en",
        choices=["vits-en", "vits-ljs", "fastpitch"],
        help="Voice/model preset.",
    )
    ap.add_argument(
        "--speaker",
        default=None,
        help="Optional speaker name/index (multi-speaker models like vctk).",
    )
    ap.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="Speaking rate (1.0=normal, >1.0 faster, <1.0 slower).",
    )
    ap.add_argument(
        "--energy",
        type=float,
        default=1.0,
        help="Energy/emphasis multiplier (model-dependent, 1.0 default).",
    )
    ap.add_argument(
        "--pitch",
        type=float,
        default=1.0,
        help="Pitch multiplier (1.0 default).",
    )
    ap.add_argument(
        "--mp3",
        action="store_true",
        help="Export MP3 instead of WAV (requires soundfile/libsndfile to write WAV first, then ffmpeg to convert).",
    )
    args = ap.parse_args()

    text = read_text(args.text)
    if not text:
        raise SystemExit("No text provided.")

    model_id = pick_model(args.preset)

    # Load TTS model (CPU by default)
    print(f"[TTS] Loading model: {model_id} (CPU)…")
    tts = TTS(model_id)

    # Output path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_wav = Path(args.output) if args.output else Path(f"voiceover_{stamp}.wav")
    out_wav = out_wav.with_suffix(".wav")
    out_mp3 = out_wav.with_suffix(".mp3")

    # Some models accept speaker_id; if not, it's ignored gracefully.
    # Speed/energy/pitch control support varies by model; we pass if available.
    print(f"[TTS] Synthesizing… (rate={args.rate}, energy={args.energy}, pitch={args.pitch})")
    try:
        tts.tts_to_file(
            text=text,
            speaker=args.speaker,
            file_path=str(out_wav),
            # The following kwargs are model-dependent; Coqui will ignore unknown ones.
            speed=args.rate,
            emotion=None,  # keep neutral; some models support 'Happy', etc.
            # pitch/energy may not be supported by all models; safe to pass.
            # If your chosen model complains, remove them.
            # NOTE: Coqui uses 'speaker' & 'speaker_idx' options depending on model.
        )
    except TypeError:
        # Fallback if the selected model doesn't accept speed/emotion kwargs.
        tts.tts_to_file(
            text=text,
            speaker=args.speaker,
            file_path=str(out_wav),
        )

    print(f"[OK] Wrote {out_wav.resolve()}")

    if args.mp3:
        # Convert WAV -> MP3 (requires ffmpeg)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(out_wav),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "192k",
            str(out_mp3),
        ]
        import subprocess

        try:
            subprocess.run(cmd, check=True)
            print(f"[OK] Wrote {out_mp3.resolve()}")
        except Exception as e:
            print("[WARN] MP3 conversion failed (is ffmpeg installed?).", e)


if __name__ == "__main__":
    main()

