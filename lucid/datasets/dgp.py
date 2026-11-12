from ._loaders import _load_dataset


def extreme_tail():
    """Load the extreme-tail synthetic dataset."""
    return _load_dataset("dgp/extreme_tail.csv", target="target")


def regime_dependent():
    """Load the regime-dependent synthetic dataset."""
    return _load_dataset("dgp/regime_dependent.csv", target="target")


def strong_conditional_dependence():
    """Load the strong conditional dependence synthetic dataset."""
    return _load_dataset(
        "dgp/strong_conditional_dependence.csv",
        target="target",
    )


def heteroskedastic_conditional_dependence():
    """Load the heteroskedastic conditional dependence synthetic dataset."""
    return _load_dataset(
        "dgp/heteroskedastic_conditional_dependence.csv",
        target="target",
    )


def interaction_conditional_dependence():
    """Load the interaction conditional dependence synthetic dataset."""
    return _load_dataset(
        "dgp/interaction_conditional_dependence.csv",
        target="target",
    )