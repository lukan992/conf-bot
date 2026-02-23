import numpy as np
import torch
from typing import List, Tuple

def compute_embeddings_nemo(
    model,
    audio_int16: np.ndarray,
    sr: int,
    windows: List[Tuple[float, float]],
    slice_waveform_int16,
    window_sec: float = 2.0,
    batch_size: int = 128,
    l2_normalize: bool = True,
):
    """
    Returns:
      embeddings: np.ndarray shape (N, D)
      meta: list of (t0, t1) for each embedding row
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    T = int(round(window_sec * sr))  # samples per window

    def iter_batches(items, bs):
        for i in range(0, len(items), bs):
            yield items[i:i+bs]

    embs = []
    meta = []

    with torch.no_grad():
        for batch_windows in iter_batches(windows, batch_size):
            B = len(batch_windows)
            batch = np.zeros((B, T), dtype=np.float32)

            for j, (t0, t1) in enumerate(batch_windows):
                x = slice_waveform_int16(audio_int16, sr, t0, t1).astype(np.float32) / 32768.0

                # на всякий случай (должно быть ровно T, если окна корректно сформированы)
                if len(x) >= T:
                    batch[j] = x[:T]
                else:
                    batch[j, :len(x)] = x

            sig = torch.from_numpy(batch).to(device)
            sig_len = torch.full((B,), T, device=device, dtype=torch.long)

            logits, emb = model(input_signal=sig, input_signal_length=sig_len)
            emb = emb.detach().cpu().numpy()  # (B, 192)

            embs.append(emb)
            meta.extend(batch_windows)

    embeddings = np.concatenate(embs, axis=0)  # (N, 192)

    if l2_normalize:
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)

    return embeddings, meta