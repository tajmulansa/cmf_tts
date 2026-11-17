import os
import subprocess
from datetime import datetime

import numpy as np
import soundfile as sf
from kokoro import KPipeline

SAMPLE_RATE = 24000

text = os.environ.get("TTS_TEXT", "").strip()
if not text:
    with open("script.txt", encoding="utf-8") as f:
        text = f.read().strip()
if not text:
    raise SystemExit("No text given and script.txt is empty.")

voice = os.environ.get("TTS_VOICE", "af_heart") or "af_heart"
speed = float(os.environ.get("TTS_SPEED", "1.0") or 1.0)

# First letter of the voice = language: a=US English, b=UK English, h=Hindi
pipeline = KPipeline(lang_code=voice[0])

pause = np.zeros(int(SAMPLE_RATE * 0.25), dtype=np.float32)
chunks = []
for _, _, audio in pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+"):
    if audio is None:
        continue
    if hasattr(audio, "numpy"):
        audio = audio.numpy()
    chunks.append(audio.astype(np.float32))
    chunks.append(pause)

if not chunks:
    raise SystemExit("No audio generated.")

os.makedirs("output", exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
wav_path = f"output/voiceover-{stamp}.wav"
mp3_path = f"output/voiceover-{stamp}.mp3"

sf.write(wav_path, np.concatenate(chunks), SAMPLE_RATE)
subprocess.run(
    ["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path, "-b:a", "192k", mp3_path],
    check=True,
)
os.remove(wav_path)
print(f"Saved {mp3_path}")
