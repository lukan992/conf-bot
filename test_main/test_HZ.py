from VAD import run_vad, merge_gap_dict, filter_min_speech_dict
from sliding_windows import make_sliding_windows_from_speech, slice_waveform_int16, int16_to_float32
from titanet import compute_embeddings_nemo
from HDBlabels import cluster_hdbscan_cosine_precomputed

import nemo.collections.asr as nemo_asr
import scipy.io.wavfile as wav
import numpy as np
from collections import defaultdict
from pathlib import Path


NEMO_PATH = "data/models/titanet_large/speakerverification_en_titanet_large.nemo"
audio_path = 'data/audio/test_conf1_16k.wav'
sr, audio = wav.read(audio_path)

if audio.ndim == 2:
        audio = audio.astype(np.int32).mean(axis=1)
        audio = np.clip(np.rint(audio), -32768, 32767).astype(np.int16)

try:
    speech_segments = run_vad(audio=audio, sr=sr, trash=0.25)
except FileNotFoundError:
    print(f'не найден файл в пути {audio_path}')

speech_seg_merged = merge_gap_dict(speech_segments, 0.3)
speech_seg_merg_min = filter_min_speech_dict(speech_seg_merged, 0.5)

windows = make_sliding_windows_from_speech(speech_seg_merg_min, 2.0, 0.5)

model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(NEMO_PATH)

print("sr:", sr)
print("audio:", audio.dtype, audio.shape, "seconds:", len(audio)/sr)

print("speech_segments:", len(speech_seg_merg_min))
if speech_seg_merg_min:
    print("first speech seg:", speech_seg_merg_min[0])

print("windows:", len(windows))
if windows:
    print("first window:", windows[0], "dur:", windows[0][1]-windows[0][0])

embeddings, meta = compute_embeddings_nemo(
    model=model,
    audio_int16=audio,
    sr=sr,
    windows=windows,
    slice_waveform_int16=slice_waveform_int16,
    window_sec=2.0,
    batch_size=128,
)

print("embeddings:", embeddings.shape)  # (N, 192)
print("first meta:", meta[0])

labels_hdb = cluster_hdbscan_cosine_precomputed(embeddings,
    min_cluster_size=20,
    min_samples=10,
)

segments_window_level = [((t0, t1), int(lbl)) for (t0, t1), lbl in zip(meta, labels_hdb)]


def merge_intervals(intervals: list[tuple[float, float]], collar: float = 0.0) -> list[tuple[float, float]]:
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda x: x[0])
    merged: list[tuple[float, float]] = [intervals[0]]

    for start, end in intervals[1:]:
        cur_start, cur_end = merged[-1]
        if start <= (cur_end + collar):
            merged[-1] = (cur_start, max(cur_end, end))
        else:
            merged.append((start, end))
    return merged


def segments_to_rttm_lines(
    segments: list[tuple[tuple[float, float], int]],
    recording_id: str,
    speaker_prefix: str = "spk_",
    include_noise: bool = False,
    merge_same_label: bool = True,
    merge_collar: float = 0.0,
    min_duration: float = 0.0,
) -> list[str]:
    by_label: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for (t_start, t_end), label in segments:
        t_start = float(t_start)
        t_end = float(t_end)
        if t_end <= t_start:
            continue
        if label == -1 and not include_noise:
            continue
        by_label[label].append((t_start, t_end))

    rttm_rows: list[tuple[float, str]] = []
    for label, intervals in by_label.items():
        clean_intervals = merge_intervals(intervals, collar=merge_collar) if merge_same_label else sorted(intervals, key=lambda x: x[0])
        speaker = f"{speaker_prefix}{label}" if label != -1 else f"{speaker_prefix}noise"
        for start, end in clean_intervals:
            duration = end - start
            if duration < min_duration:
                continue
            line = f"SPEAKER {recording_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>"
            rttm_rows.append((start, line))

    rttm_rows.sort(key=lambda x: x[0])
    return [line for _, line in rttm_rows]


def write_rttm(
    segments: list[tuple[tuple[float, float], int]],
    rttm_path: str,
    recording_id: str,
) -> None:
    lines = segments_to_rttm_lines(
        segments=segments,
        recording_id=recording_id,
        include_noise=False,
        merge_same_label=True,
        merge_collar=0.0,
        min_duration=0.01,
    )
    content = "\n".join(lines)
    if content:
        content += "\n"
    Path(rttm_path).write_text(content, encoding="utf-8")


recording_id = Path(audio_path).stem
rttm_output_path = f"data/rttm/{recording_id}.rttm"
Path("data/rttm").mkdir(parents=True, exist_ok=True)
write_rttm(segments_window_level, rttm_output_path, recording_id)
print(f"RTTM saved: {rttm_output_path}")
