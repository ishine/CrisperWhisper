"""Backend-agnostic token-loop detection.

Post-hoc loop detection (:func:`find_token_loop`) scans a finished token
sequence for repeated n-gram patterns.  It is used by *every* backend --
the CTranslate2 repair/blocking strategies in
:mod:`crisperwhisper.hallucination`, the Transformers backend's
``generate_with_repair``, and the temperature fallback -- so it lives in
this dependency-free module rather than in ``hallucination.py``, which
imports ``ctranslate2`` at module level (issue #57).
"""

from __future__ import annotations

DEFAULT_REPAIR_THRESHOLDS: dict[int, int] = {1: 8, 2: 8, 3: 4, 4: 3, 5: 3}
"""Per-ngram-size repetition thresholds for ``generate_with_repair``.

Keys are n-gram sizes, values are the number of consecutive repetitions
that trigger a repair.  Larger n-grams use lower thresholds because
long repeated phrases are almost never genuine speech.
"""


def find_token_loop(
    ids: list[int],
    min_ngram: int = 1,
    max_ngram: int = 5,
    reps: int | dict[int, int] = 8,
) -> tuple[int, tuple[int, ...]] | None:
    """Find the earliest position where consecutive n-grams repeat.

    Parameters
    ----------
    ids : list[int]
        Token ID sequence to scan.
    min_ngram, max_ngram : int
        Range of n-gram sizes to check.
    reps : int or dict[int, int]
        If an ``int``, the same threshold is used for all n-gram sizes.
        If a ``dict``, maps ``{ngram_size: threshold}`` — n-gram sizes
        not in the dict are skipped.

    Returns ``(loop_start_index, ngram_tuple)`` or ``None``.
    """
    thresholds: dict[int, int]
    if isinstance(reps, int):
        thresholds = {n: reps for n in range(min_ngram, max_ngram + 1)}
    else:
        thresholds = reps

    for i in range(len(ids)):
        for n, needed_reps in thresholds.items():
            if n < min_ngram or n > max_ngram:
                continue
            needed = n * needed_reps
            if i + needed > len(ids):
                continue
            gram = tuple(ids[i : i + n])
            if all(
                tuple(ids[i + r * n : i + (r + 1) * n]) == gram
                for r in range(1, needed_reps)
            ):
                return i, gram
    return None
