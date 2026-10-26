import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    N = len(seqs)
    if N == 0:
        L = max_len if max_len is not None else 0
        return np.empty((0, L), dtype=int)

    L = max_len if max_len is not None else max(len(seq) for seq in seqs)

    # Pre-allocate output matrix filled with pad_value
    padded = np.full((N, L), pad_value, dtype=int)

    # Fill each row up to min(len(seq), L) for safe padding/truncation
    for i, seq in enumerate(seqs):
        length = min(len(seq), L)
        padded[i, :length] = seq[:length]

    return padded