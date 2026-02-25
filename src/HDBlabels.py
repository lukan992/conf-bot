import numpy as np
import hdbscan
import os
import threading


def _patch_multiprocess_resource_tracker_py312() -> None:
    """
    Work around multiprocess 0.70.18 teardown bug on Python 3.12:
    ResourceTracker._stop_locked unconditionally calls
    RLock._recursion_count(), but this private method is not present.
    """
    if hasattr(threading.RLock(), "_recursion_count"):
        return

    try:
        from multiprocess import resource_tracker as resource_tracker_mod
    except Exception:
        return

    tracker_cls = resource_tracker_mod.ResourceTracker
    if getattr(tracker_cls, "_py312_rlock_patch_applied", False):
        return

    def _stop_locked_compat(
        self,
        close=os.close,
        waitpid=os.waitpid,
        waitstatus_to_exitcode=os.waitstatus_to_exitcode,
    ):
        recursion_count = getattr(self._lock, "_recursion_count", None)
        if callable(recursion_count) and recursion_count() > 1:
            return self._reentrant_call_error()
        if self._fd is None or self._pid is None:
            return
        close(self._fd)
        self._fd = None
        waitpid(self._pid, 0)
        self._pid = None

    tracker_cls._stop_locked = _stop_locked_compat
    tracker_cls._py312_rlock_patch_applied = True


_patch_multiprocess_resource_tracker_py312()


def cluster_hdbscan_cosine_precomputed(
    embeddings: np.ndarray,
    min_cluster_size: int = 20,
    min_samples: int | None = 10,
):
    """
    embeddings: (N, D), L2-normalized
    returns labels: (N,), with possible -1 (noise)
    """
    # cosine similarity (float64 for hdbscan)
    E = np.asarray(embeddings, dtype=np.float64)
    sim = E @ E.T
    sim = np.clip(sim, -1.0, 1.0)

    dist = 1.0 - sim
    dist = dist.astype(np.float64, copy=False)

    # clean up numerical issues
    np.fill_diagonal(dist, 0.0)
    dist = 0.5 * (dist + dist.T)

    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(dist)
    return labels


def merge_consecutive_by_label(segments):
    if not segments:
        return []

    merged = []
    cur_start, cur_end, cur_label = segments[0]

    for start, end, label in segments[1:]:
        if label == cur_label:
            cur_end = max(cur_end, end)  # тянем конец до последнего куска
        else:
            merged.append((cur_start, cur_end, cur_label))
            cur_start, cur_end, cur_label = start, end, label

    merged.append((cur_start, cur_end, cur_label))
    return merged
