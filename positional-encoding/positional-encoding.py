import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    # Write code here
    pos = np.arange(seq_len)[:,np.newaxis] # shape : (seq_len,1)
    i = np.arange(0,d_model,2)  # equivalent to half of d_model
    
    div_term = 1.0/(base ** (i/d_model))

    angles = pos * div_term  # shape : (seq_len,half)

    pe = np.zeros((seq_len, d_model)) # shape : (seq_len,d_model)

    pe[:,0::2] = np.sin(angles) 
    pe[:, 1::2] = np.cos(angles[:, :d_model // 2])

    return pe
    