import nemo.collections.asr as nemo_asr
import numpy as np
import scipy.io.wavfile as wav
from torch import threshold
from dotenv import load_dotenv
import os
from pathlib import Path


if __package__:
    from .VAD import run_vad, merge_gap_dict, filter_min_speech_dict
    from .sliding_windows import make_sliding_windows_from_speech, slice_waveform_int16, int16_to_float32
    from .titanet import compute_embeddings_nemo
    from .TextFromAudio import get_stt
    from .HDBlabels import cluster_hdbscan_cosine_precomputed
else:
    from VAD import run_vad, merge_gap_dict, filter_min_speech_dict
    from sliding_windows import make_sliding_windows_from_speech, slice_waveform_int16, int16_to_float32
    from titanet import compute_embeddings_nemo
    from TextFromAudio import get_stt
    from HDBlabels import cluster_hdbscan_cosine_precomputed, merge_consecutive_by_label


def overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def attach_text_to_diar(diar_segments, words):
    results = []

    for d_start, d_end, spk in diar_segments:
        segment_words = []

        for w in words:
            if overlap(w["start"], w["end"], d_start, d_end) > 0:
                segment_words.append(w)

        if not segment_words:
            continue

        text = " ".join(w["word"] for w in segment_words)

        new_start = min(w["start"] for w in segment_words)
        new_end = max(w["end"] for w in segment_words)

        results.append({
            "start": new_start,
            "end": new_end,
            "speaker": spk,
            "text": text
        })

    return results


def text_diar_from_audio(
    audio_path: str,
    nemo_path:str,
    threshold_vad: float,
    gap: float,
    win_l: float,
    win_s: float,
    min_speech: float
):
    #thrashold_vad - влияет на количество отрезков с речью. чем меньше, тем меньше отрезков (шумы могут стать речью)
    #gap - максимально допустимая пауза между двумя speech-сегментами, при которой они считаются одним непрерывным участком речи.
    #min_speech - минимальная допустимая длительность speech-сегмента (в секундах). Функция удаляет все участки речи, которые короче этого порога. 
    #win_l - размер скользящего окна в секундах
    #win_s - размер шага в секундах
    sr, audio = wav.read(audio_path)

    if audio.ndim == 2:
        audio = audio.astype(np.int32).mean(axis=1)
        audio = np.clip(np.rint(audio), -32768, 32767).astype(np.int16)

    try:
        speech_segments = run_vad(audio=audio, sr=sr, trash=threshold_vad)
    except FileNotFoundError:
        print(f'не найден файл в пути {audio_path}')

    speech_seg_merged = merge_gap_dict(speech_segments, gap=gap)
    speech_seg_merg_min = filter_min_speech_dict(speech_seg_merged, min_speech)

    windows = make_sliding_windows_from_speech(speech_seg_merg_min, win_l, win_s)

    model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(nemo_path)

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

    diar_segments = [(t0, t1, int(lbl)) for (t0, t1), lbl in zip(meta, labels_hdb)]
    
    text_from_audio = get_stt(audio_path)

    segments_window_level = merge_consecutive_by_label(diar_segments)

    aligned_words = attach_text_to_diar(segments_window_level, text_from_audio)

    return aligned_words


def save_segments(
    segments,
    output_dir: str,
    filename: str,
    file_type: str,
    file_id: str = "audio",
    skip_unknown: bool = False,
):
    """
    segments: list of dicts with keys:
        - start
        - end
        - speaker
        - text (optional for rttm)

    output_dir: путь к папке
    filename: имя файла без расширения
    file_type: 'txt' или 'rttm'
    file_id: используется только для rttm
    skip_unknown: если True — сегменты с speaker=-1 не сохраняются
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_type = file_type.lower()

    if file_type not in {"txt", "rttm"}:
        raise ValueError("file_type must be 'txt' or 'rttm'")

    output_path = output_dir / f"{filename}.{file_type}"

    with open(output_path, "w", encoding="utf-8") as f:

        for seg in segments:
            start = float(seg["start"])
            end = float(seg["end"])
            speaker = seg["speaker"]

            if skip_unknown and speaker == -1:
                continue

            if file_type == "txt":
                text = seg.get("text", "")
                line = f"[{start:.2f} - {end:.2f}] Speaker {speaker}: {text}\n"
                f.write(line)

            elif file_type == "rttm":
                duration = end - start
                speaker_label = (
                    "speaker_unknown"
                    if speaker == -1
                    else f"speaker_{speaker}"
                )

                line = (
                    f"SPEAKER {file_id} 1 "
                    f"{start:.3f} {duration:.3f} "
                    f"<NA> <NA> {speaker_label} <NA> <NA>\n"
                )

                f.write(line)

    return output_path


if __name__ == "__main__":
    load_dotenv()

    audio_path = os.getenv("AUDIO_PATH")
    nemo_path = os.getenv("NEMO_PATH")

    threshold_vad = float(os.getenv("THRESHOLD_VAD", 0.5))
    gap = float(os.getenv("GAP", 0.3))
    win_l = float(os.getenv("WIN_L", 2.0))
    win_s = float(os.getenv("WIN_S", 0.5))
    min_speech = float(os.getenv("MIN_SPEECH", 0.5)) 
    aligned_words = text_diar_from_audio(
        audio_path=audio_path,
        nemo_path=nemo_path,
        threshold_vad=threshold_vad,
        gap=gap,
        win_l=win_l,
        win_s=win_s,
        min_speech=min_speech
    )
    # for w in aligned_words:
    #     print(w)
    save_segments(aligned_words, "data/output", "aligned_words", "txt")
