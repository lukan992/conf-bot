import numpy as np
from typing import List, Dict, Tuple


def make_sliding_windows_from_speech(
    speech_segments: List[Dict],
    window_sec: float = 2.0,
    step_sec: float = 0.5,
) -> List[Tuple[float, float]]:
    """
    speech_segments: list of dicts with keys {"start","end"} in seconds (original timeline)
    returns: list of (t0, t1) windows in seconds (original timeline)
    """
    windows: List[Tuple[float, float]] = []

    for seg in speech_segments:
        S = float(seg["start"])
        E = float(seg["end"])
        t = S

        # only windows fully inside [S, E]
        while (t + window_sec) <= E:
            windows.append((t, t + window_sec))
            t += step_sec

    return windows


def slice_waveform_int16(
    audio_int16: np.ndarray,
    sr: int,
    t0: float,
    t1: float,
) -> np.ndarray:
    """
    Returns int16 audio chunk corresponding to [t0, t1) seconds.
    """
    i0 = int(round(t0 * sr))
    i1 = int(round(t1 * sr))
    i0 = max(0, min(i0, len(audio_int16)))
    i1 = max(0, min(i1, len(audio_int16)))
    return audio_int16[i0:i1]


def int16_to_float32(audio_int16: np.ndarray) -> np.ndarray:
    """
    Convert int16 PCM to float32 in [-1, 1).
    """
    return audio_int16.astype(np.float32) / 32768.0


