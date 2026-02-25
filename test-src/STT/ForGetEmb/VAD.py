from ten_vad import TenVad
import numpy as np
import scipy.io.wavfile as wav


def run_vad(audio: np.ndarray, sr: int, trash: float = 0.5):
    # sr, audio = wav.read(audio_path)  audio в int16, 16 kHz
    # if audio.ndim == 2:
    #     audio = audio.astype(np.int32).mean(axis=1)
    #     audio = np.clip(np.rint(audio), -32768, 32767).astype(np.int16)

    # -----------------------------
    # 2) TEN-VAD frame inference
    # -----------------------------
    vad = TenVad(hop_size=256, threshold=trash)

    frames = []  # store (prob, flag)
    num_frames = len(audio) // vad.hop_size

    for k in range(num_frames):
        i = k * vad.hop_size
        frame = audio[i : i + vad.hop_size]
        prob, flag = vad.process(frame)  # returns (probability, flag) :contentReference[oaicite:1]{index=1}
        frames.append((prob, int(flag)))

    # -----------------------------
    # 3) Convert frame flags -> speech intervals
    # -----------------------------
    speech_segments = []  # list of dicts with start/end (+ optional stats)

    in_speech = False
    start_frame = None
    prob_sum = 0.0
    n_prob = 0

    for idx, (prob, flag) in enumerate(frames):
        if flag == 1 and not in_speech:
            in_speech = True
            start_frame = idx
            prob_sum = float(prob)
            n_prob = 1

        elif flag == 1 and in_speech:
            prob_sum += float(prob)
            n_prob += 1

        elif flag == 0 and in_speech:
            end_frame = idx  # exclusive
            t_start = (start_frame * vad.hop_size) / sr
            t_end = (end_frame * vad.hop_size) / sr

            speech_segments.append({
                "start": t_start,
                "end": t_end,
                "flag": 1,
                "mean_prob": prob_sum / max(n_prob, 1),
                "n_frames": n_prob,
            })

            in_speech = False
            start_frame = None

    # close tail segment if file ended during speech
    if in_speech and start_frame is not None:
        end_frame = len(frames)
        t_start = (start_frame * vad.hop_size) / sr
        t_end = (end_frame * vad.hop_size) / sr

        speech_segments.append({
            "start": t_start,
            "end": t_end,
            "flag": 1,
            "mean_prob": prob_sum / max(n_prob, 1),
            "n_frames": n_prob,
        })
    return speech_segments


from typing import List, Dict

def merge_gap_dict(segments: List[Dict], gap: float) -> List[Dict]:
    """
    Merge adjacent speech segments if gap between them < gap seconds.
    Keeps stats: n_frames sum; mean_prob as weighted mean by n_frames.
    Assumes segments are sorted by start time and all are speech (flag=1).
    """
    if not segments:
        return []

    merged = []
    cur = dict(segments[0])  # copy

    for seg in segments[1:]:
        # gap between current end and next start
        if seg["start"] <= cur["end"] + gap:
            # merge
            old_frames = cur.get("n_frames", 0)
            new_frames = seg.get("n_frames", 0)

            # update time bounds
            cur["end"] = max(cur["end"], seg["end"])

            # update n_frames
            cur["n_frames"] = old_frames + new_frames

            # weighted mean probability (if present)
            if "mean_prob" in cur and "mean_prob" in seg and (old_frames + new_frames) > 0:
                cur["mean_prob"] = (
                    cur["mean_prob"] * old_frames + seg["mean_prob"] * new_frames
                ) / (old_frames + new_frames)

            cur["flag"] = 1
        else:
            merged.append(cur)
            cur = dict(seg)

    merged.append(cur)
    return merged


def filter_min_speech_dict(segments: List[Dict], min_speech: float) -> List[Dict]:
    """
    Keep only speech segments with duration >= min_speech seconds.
    segments: list of dicts with "start", "end"
    """
    out = []
    for seg in segments:
        if (seg["end"] - seg["start"]) >= min_speech:
            out.append(seg)
    return out


if __name__ == "__main__":
    audio_path = ''
    speech_segments = run_vad(audio_path)
    speech_seg_merged = merge_gap_dict(speech_segments, 0.3)
    speech_seg_merg_min = filter_min_speech_dict(speech_seg_merged, 0.5)
