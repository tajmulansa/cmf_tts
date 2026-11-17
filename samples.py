"""Render the same CMF-style line in many Kokoro voices for comparison."""
import os
import subprocess

import numpy as np
import soundfile as sf
from kokoro import KPipeline

SAMPLE_RATE = 24000

TEXT = (
    "Here's something that sounds fake, but isn't.\n"
    "Your stomach makes a brand new lining every few days.\n"
    "Because the acid inside it is strong enough to dissolve metal.\n"
    "So right now, your body is quietly rebuilding itself, just so it doesn't digest itself."
)

VOICES = os.environ.get(
    "VOICES",
    "af_heart,af_bella,af_nicole,af_sarah,af_nova,am_michael,am_fenrir,am_puck,"
    "am_adam,am_echo,bf_emma,bf_isabella,bm_george,bm_fable,bm_lewis,bm_daniel",
).split(",")

pause = np.zeros(int(SAMPLE_RATE * 0.25), dtype=np.float32)
pipelines = {}
os.makedirs("samples", exist_ok=True)

for voice in [v.strip() for v in VOICES if v.strip()]:
    try:
        lang = voice[0]
        if lang not in pipelines:
            pipelines[lang] = KPipeline(lang_code=lang)
        chunks = []
        for _, _, audio in pipelines[lang](TEXT, voice=voice, speed=1.0, split_pattern=r"\n+"):
            if audio is None:
                continue
            if hasattr(audio, "numpy"):
                audio = audio.numpy()
            chunks.append(audio.astype(np.float32))
            chunks.append(pause)
        wav = f"samples/{voice}.wav"
        sf.write(wav, np.concatenate(chunks), SAMPLE_RATE)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-b:a", "160k", f"samples/{voice}.mp3"],
            check=True,
        )
        os.remove(wav)
        print(f"OK   {voice}")
    except Exception as e:  # keep going if one voice fails
        print(f"FAIL {voice}: {e}")
