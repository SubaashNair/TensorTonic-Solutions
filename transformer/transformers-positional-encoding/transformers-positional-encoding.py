import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    pe = np.zeros((seq_length, d_model))
    positions = np.arange(seq_length).reshape(-1, 1)  # (seq_length, 1)
    i = np.arange(0, d_model, 2)                       # even indices: 0, 2, 4, ...
    div_term = np.exp(i * (-np.log(10000.0) / d_model))  # (d_model/2,)

    pe[:, 0::2] = np.sin(positions * div_term)
    pe[:, 1::2] = np.cos(positions * div_term)

    return pe