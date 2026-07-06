"""Time-series-safe validation splits.

Standard k-fold cross-validation shuffles rows and leaks future information
into the training set. Walk-forward splits keep every test block strictly
after its train block, which is the only valid way to evaluate a trading
signal out-of-sample.
"""

from __future__ import annotations

from typing import Iterator

import pandas as pd


def walk_forward_splits(
    index: pd.Index, train_size: int, test_size: int, step: int | None = None
) -> Iterator[tuple[pd.Index, pd.Index]]:
    """Yield (train_index, test_index) pairs walking forward through `index`.

    Each test block immediately follows its train block with zero overlap.
    `step` controls how far the window advances between folds; defaults to
    `test_size` (non-overlapping test blocks).
    """
    step = step or test_size
    n = len(index)
    start = 0
    while start + train_size + test_size <= n:
        train_idx = index[start : start + train_size]
        test_idx = index[start + train_size : start + train_size + test_size]
        yield train_idx, test_idx
        start += step
