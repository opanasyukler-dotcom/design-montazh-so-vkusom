"""Голосовой клон: озвучка любого текста голосом из образца (XTTS-v2, локально, CPU).

Модель капризная и иногда «съезжает» (выдумывает слова, падает тон), поэтому
каждое предложение генерируется несколькими дублями, а лучший выбирается
автоматически: сверка с текстом через Whisper + близость тона к образцу.

Установка (один раз):
    pip install coqui-tts "torch==2.8.*" "torchaudio==2.8.*" faster-whisper librosa \
        --extra-index-url https://download.pytorch.org/whl/cpu

Использование:
    python tools/voice_clone.py --ref голос.wav --text "Текст для озвучки" --out voice.wav
    python tools/voice_clone.py --ref видео.mp4 --text-file сценарий.txt --out voice.wav

Образец: 10–60 с чистой речи без музыки. Можно передать видео — звук извлечётся сам.
"""
import argparse
import difflib
import os
import re
import subprocess
import tempfile

import librosa
import numpy as np
import soundfile as sf

os.environ.setdefault("COQUI_TOS_AGREED", "1")
SR = 24000


def prepare_ref(src, dst):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", src, "-ac", "1", "-ar", str(SR), "-af",
         "highpass=f=80,lowpass=f=11000,afftdn=nf=-25,loudnorm=I=-18:TP=-2", dst],
        check=True,
    )


def median_f0(y, sr):
    f0, _, _ = librosa.pyin(y, fmin=60, fmax=500, sr=sr)
    f0 = f0[~np.isnan(f0)]
    return float(np.median(f0)) if len(f0) else 0.0


def norm(s):
    return re.sub(r"[^а-яa-z0-9 ]", "", s.lower().replace("ё", "е")).split()


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?…])\s+", text.strip()) if s.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="образец голоса (wav/mp3/mp4)")
    ap.add_argument("--text")
    ap.add_argument("--text-file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--takes", type=int, default=6, help="дублей на предложение")
    ap.add_argument("--speed", type=float, default=1.1)
    ap.add_argument("--pause", type=float, default=0.25, help="пауза между фразами, с")
    a = ap.parse_args()
    text = a.text or open(a.text_file, encoding="utf-8").read()

    from faster_whisper import WhisperModel
    from TTS.api import TTS

    tmp = tempfile.mkdtemp()
    ref = os.path.join(tmp, "ref.wav")
    prepare_ref(a.ref, ref)
    y_ref, _ = librosa.load(ref, sr=16000)
    ref_f0 = median_f0(y_ref, 16000)
    print(f"тон образца: {ref_f0:.0f} Гц")

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
    asr = WhisperModel("small", device="cpu", compute_type="int8")

    pieces = []
    for i, sent in enumerate(split_sentences(text), 1):
        best = None
        for take in range(a.takes):
            wav = np.array(tts.tts(text=sent, speaker_wav=ref, language="ru",
                                   temperature=0.5, speed=a.speed), dtype=np.float32)
            y16 = librosa.resample(wav, orig_sr=SR, target_sr=16000)
            heard = " ".join(s.text for s in asr.transcribe(y16, language="ru")[0])
            match = difflib.SequenceMatcher(None, norm(sent), norm(heard)).ratio()
            f0 = median_f0(y16, 16000)
            score = match - abs(f0 - ref_f0) / ref_f0
            print(f"  [{i}] дубль {take + 1}: текст {match:.2f}, тон {f0:.0f} Гц")
            if best is None or score > best[0]:
                best = (score, wav)
            if match > 0.95 and abs(f0 - ref_f0) < 0.1 * ref_f0:
                break
        trimmed, _ = librosa.effects.trim(best[1], top_db=25)
        pieces += [trimmed, np.zeros(int(a.pause * SR), dtype=np.float32)]

    sf.write(a.out, np.concatenate(pieces), SR)
    print("готово:", a.out)


if __name__ == "__main__":
    main()
