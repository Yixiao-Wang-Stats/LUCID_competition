import numpy as np
from . import loss


# For feature j observation i.
def lucid_single_observation_single_feature(
    x: np.ndarray,
    y,
    replacement_sample: np.ndarray,
    n_resamples_batch: int,
    loss_type: str,
    model: callable,
    feature_idx: int,
    baseline_loss=None,
) -> float:
    """
    Compute LUCID for one observation and one feature.

    replacement_sample has shape (n_resamples,).
    """
    return lucid_single_observation_group(
        x=x,
        y=y,
        replacement_values=np.asarray(replacement_sample)[None, :],
        n_resamples_batch=n_resamples_batch,
        loss_type=loss_type,
        model=model,
        feature_indices=[feature_idx],
        baseline_loss=baseline_loss,
    )

# For feature group S and observation i.
def lucid_single_observation_group(
    x: np.ndarray,
    y,
    replacement_values: np.ndarray,
    n_resamples_batch: int,
    loss_type: str,
    model: callable,
    feature_indices,
    baseline_loss=None,
)->float:
    """
    Compute LUCID for one observation and one feature group.

    replacement_values has shape:
        (group_size, n_resamples)
    """

    x = np.asarray(x)
    replacement_values = np.asarray(replacement_values)
    feature_indices = np.asarray(feature_indices,dtype=int)

    if x.ndim != 1:
        raise ValueError("x must be a 1D feature vector.")

    if feature_indices.ndim != 1:
        raise ValueError("feature_indices must be a 1D array.")

    if len(feature_indices) == 0:
        raise ValueError("feature_indices must not be empty.")

    if len(np.unique(feature_indices)) != len(feature_indices):
        raise ValueError("feature_indices must not contain duplicates.")

    if np.any(feature_indices < 0) or np.any(feature_indices >= x.shape[0]):
        raise ValueError("feature_indices contains an out-of-range feature index.")

    if replacement_values.ndim != 2:
        raise ValueError("replacement_values must be a 2D array.")

    if replacement_values.shape[0] != len(feature_indices):
        raise ValueError(
            "replacement_values must have one row for every feature in feature_indices."
        )

    if n_resamples_batch <= 0:
        raise ValueError("n_resamples_batch must be positive.")

    n_resamples = replacement_values.shape[1]

    if n_resamples == 0:
        raise ValueError("replacement_values must not be empty.")

    if baseline_loss is None:
        baseline_prediction = model(x.reshape(1,-1))
        baseline_loss = loss.prediction_loss(baseline_prediction,y,loss_type,).item()

    total_loss = 0.0

    for start in range(
        0,
        n_resamples,
        n_resamples_batch,
    ):
        group_prime = replacement_values[:,start:start + n_resamples_batch,]
        current_batch_size = group_prime.shape[1]
        x_prime = np.repeat(
            x.reshape(1,-1),
            current_batch_size,
            axis=0,
        ).astype(float)
        x_prime[:,feature_indices] = group_prime.T  # (current_batch_size, group_size)
        batch_loss = loss.prediction_loss(model(x_prime),y,loss_type,)

        total_loss += np.sum(batch_loss)

    return total_loss / n_resamples - baseline_loss
