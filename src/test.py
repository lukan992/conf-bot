from VAD import run_vad, merge_gap_dict, filter_min_speech_dict
from sliding_windows import make_sliding_windows_from_speech, slice_waveform_int16, int16_to_float32
from titanet import compute_embeddings_nemo


audio_path = 'data/audio/audio-test.wav'

try:
    speech_segments = run_vad(audio_path)
except FileNotFoundError:
    print(f'не найден файл в пути {audio_path}')

speech_seg_merged = merge_gap_dict(speech_segments, 0.3)
speech_seg_merg_min = filter_min_speech_dict(speech_seg_merged, 0.5)

windows = make_sliding_windows_from_speech(speech_seg_merg_min, 2.0, 0.5)
