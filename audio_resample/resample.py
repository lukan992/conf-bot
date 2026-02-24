from pathlib import Path

import numpy as np
import soundfile as sf
import librosa


def convert_wav_to_16k_mono(
    in_path: str | Path,
    out_path: str | Path,
    target_sr: int = 16000,
) -> None:
    in_path = Path(in_path)
    out_path = Path(out_path)

    # Всегда читаем как float32: проще и безопаснее для ресемплинга
    audio, sr = sf.read(in_path, dtype="float32", always_2d=True)  # shape: (n, ch)
    # mono
    mono = audio.mean(axis=1)

    if sr != target_sr:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")

    # Клиппинг на всякий случай
    mono = np.clip(mono, -1.0, 1.0)

    # Пишем в PCM16
    sf.write(out_path, mono, target_sr, subtype="PCM_16")


if __name__ == "__main__":
    convert_wav_to_16k_mono("data/audio/audio-test1.wav", "data/audio/artur.wav")