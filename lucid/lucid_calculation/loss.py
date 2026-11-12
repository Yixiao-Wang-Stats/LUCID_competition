import numpy as np

def prediction_loss(prediction: np.ndarray, target: np.ndarray, loss_type: str) -> np.ndarray:
    """
    Return elementwise loss, so output has the same shape as the inputs
    loss_type contains:
    - "l2" or "squared_error": Squared error loss
    - "l1" or "absolute_error": Absolute error loss
    - "ce" or "cross_entropy": Cross entropy loss 
    """

    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    if loss_type in {"l2","mse", "squared_error"}:
        return (prediction - target) ** 2
    if loss_type in {"l1", "absolute_error"}:
        return np.abs(prediction - target)
    if loss_type in {"ce", "cross_entropy"}:
        # For binary classification
        probability = np.clip(prediction, 1e-15, 1.0 - 1e-15)
        return -(target * np.log(probability) + (1.0 - target) * np.log(1.0 - probability))
    raise ValueError(
        "loss must be one of: absolute_error, squared_error, cross_entropy."
    )