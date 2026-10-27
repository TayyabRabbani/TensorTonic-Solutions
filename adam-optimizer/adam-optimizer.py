import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    # Write code here
    param = np.array(param)
    grad = np.array(grad)
    m = np.array(m)
    v = np.array(v)
    
    m_new = (beta1 * m) + (1-beta1) * grad

    v_new = (beta2 * v) + (1-beta2) * np.square(grad)

    m_hat = m_new / (1 -(beta1**t))
    v_hat = v_new / (1-(beta2**t))

    param_new = param - (lr * (m_hat/(np.sqrt(v_hat) + eps)))

    return param_new , m_new , v_new
    