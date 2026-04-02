import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    X = np.array(X, dtype=float)
    single_dim = X.ndim == 1
    if single_dim:
        X = X.reshape(-1, 1)

    result = X.copy()
    for col in range(result.shape[1]):
        column = result[:, col]
        nan_mask = np.isnan(column)
        if not np.any(nan_mask):
            continue
        valid = column[~nan_mask]
        if len(valid) == 0:
            fill = 0.0
        elif strategy == 'mean':
            fill = np.mean(valid)
        else:
            fill = np.median(valid)
        column[nan_mask] = fill

    if single_dim:
        return result.reshape(-1)
    return result