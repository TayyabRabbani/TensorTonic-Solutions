import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    gen = rng if rng is not None else np.random.default_rng()

    # Calculate scale factor: 1 / (1 - p)
    scale = 1.0 / (1.0 - p) if p < 1.0 else 0.0

    # Generate random numbers in [0, 1)
    # Keep element if random_val >= p
    random_vals = gen.random(x_arr.shape)
    
    # Mask values: scale factor for kept elements, 0.0 for dropped elements
    mask = np.where(random_vals >= p, scale, 0.0)

    # Output is element-wise multiplication of x and the scaled mask
    output = x_arr * mask

    return output, mask