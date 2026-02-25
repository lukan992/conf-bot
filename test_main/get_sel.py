from VAD import run_vad, merge_gap_dict, filter_min_speech_dict
from sliding_windows import make_sliding_windows_from_speech, slice_waveform_int16, int16_to_float32
from titanet import compute_embeddings_nemo
from HDBlabels import cluster_hdbscan_cosine_precomputed

import nemo.collections.asr as nemo_asr
import scipy.io.wavfile as wav
import numpy as np
from collections import defaultdict
from pathlib import Path


# audio_path = 'data/audio/artur.wav'



def get_sel(audio_path: str):
    NEMO_PATH = "data/models/titanet_large/speakerverification_en_titanet_large.nemo"
    sr, audio = wav.read(audio_path)

    if audio.ndim == 2:
        audio = audio.astype(np.int32).mean(axis=1)
        audio = np.clip(np.rint(audio), -32768, 32767).astype(np.int16)

    try:
        speech_segments = run_vad(audio=audio, sr=sr, trash=0.5)
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

    labels_hdb = cluster_hdbscan_cosine_precomputed(embeddings,
        min_cluster_size=20,
        min_samples=10,
    )

    segments_window_level = [((t0, t1), int(lbl)) for (t0, t1), lbl in zip(meta, labels_hdb)]

    return segments_window_level