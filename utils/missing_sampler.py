import math
import random
from typing import List


def sample_missing_mask(num_modalities: int, xi: float) -> List[int]:
    """
    Sample a missing-modality mask r in {0,1}^M per Implementation Details:
    - Exactly 1 missing with prob xi
    - Exactly 2 missing with prob xi/2
    - Exactly 3 missing with prob xi/4
    - ... (higher-order halves each step)
    - Remaining probability assigned to complete (no missing)

    Returns a list of length M with 1 for observed, 0 for missing.
    """
    M = num_modalities
    # build geometric probabilities for k=1..M (halving each order)
    probs = []
    rem = xi
    for k in range(1, M + 1):
        p = xi / (2 ** (k - 1))
        probs.append(p)
        rem -= 0  # keep xi reference; we'll normalize below
    # sum of geometric tail up to M terms: xi * (2 - 2^(1-M)) but we simply cap to available mass
    total_geo = sum(probs)
    # rescale so that sum over k <= 1, with leftover to k=0 (complete)
    scale = min(1.0, total_geo)
    if total_geo > 0:
        probs = [p * (xi / total_geo) for p in probs]  # ensure first term ~ xi when M large
    p_complete = max(0.0, 1.0 - sum(probs))

    # categorical over k=0..M
    buckets = [p_complete] + probs
    r = random.random()
    cum = 0.0
    chosen_k = 0
    for idx, p in enumerate(buckets):
        cum += p
        if r <= cum:
            chosen_k = idx  # idx==0 => k=0, else k=idx
            break
    if chosen_k == 0:
        return [1] * M
    k = chosen_k  # number of missing modalities
    # choose k indices uniformly to set as missing
    miss_idxs = set(random.sample(range(M), k=min(k, M)))
    mask = [0 if i in miss_idxs else 1 for i in range(M)]
    return mask


def batch_sample_missing_masks(batch_size: int, num_modalities: int, xi: float) -> List[List[int]]:
    return [sample_missing_mask(num_modalities, xi) for _ in range(batch_size)]


