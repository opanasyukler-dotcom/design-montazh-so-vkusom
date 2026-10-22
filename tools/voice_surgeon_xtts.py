import os, torch
os.environ["COQUI_TOS_AGREED"] = "1"
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
text = "Добрый день! Сегодня у нас комплексная операция: абдоминопластика, липофилинг груди, подтяжка груди."
for k in range(5):
    torch.manual_seed(k)
    tts.tts_to_file(text=text, speaker_wav=["ref1.wav", "ref2.wav"], language="ru", file_path=f"take{k}.wav", temperature=0.65, speed=1.0)
