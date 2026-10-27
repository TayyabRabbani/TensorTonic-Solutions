import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    w_arr = np.asarray(w)
    g_arr = np.asarray(g)
    s_arr = np.asarray(s)
    
    s_new = (beta * s_arr) + (1 - beta) * np.square(g_arr)
    
    w_new = w_arr - (lr / (np.sqrt(s_new) + eps)) * g_arr
    
    # 3. Convert back to Python lists to match the return type hints
    return w_new.tolist(), s_new.tolist()
