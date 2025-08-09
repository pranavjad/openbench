from typing import List, Optional

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

# These imports will be implemented separately; we assume they exist.
from openbench.datasets.mrcr import get_dataset
from openbench.scorers.mrcr import mrcr_scorer


# Default bin boundaries for plotting accuracy vs input tokens (prompt + answer)
# Intervals are right-closed: 
# [4096, 8192], (8192, 16384], (16384, 32768], (32768, 65536], (65536, 131072], (131072, 262144], (262144, 524288], (524288, 1048576]
DEFAULT_MRCR_BINS: List[int] = [
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
]


@task
def mrcr(needles: int = 2, bins: Optional[List[int]] = None) -> Task:
    """Memory-Recall with Contextual Retrieval (MRCR).

    Evaluates retrieval and recall in long contexts by placing a specified
    number of "needles" (facts) in the prompt and measuring whether the
    model can correctly extract and use them.

    Args:
        needles: Number of needles in the context (allowed: 2, 4, 8). Defaults to 2.
        bins: Bin boundaries for aggregating accuracy vs total tokens
            (prompt + answer). Defaults to:
            [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576].

    Returns:
        Task configured for MRCR evaluation.
    """

    if needles not in {2, 4, 8}:
        raise ValueError("'needles' must be one of {2, 4, 8}")

    bin_edges: List[int] = bins if bins is not None else DEFAULT_MRCR_BINS

    return Task(
        dataset=get_dataset(needles=needles),
        solver=generate(),
        scorer=mrcr_scorer(needles=needles),
        name="mrcr",
        config=GenerateConfig(temperature=0.0),
    )


