"""Agents: rule-based heuristic opponent and self-play opponent pool."""
from aircombat_bench.agents.heuristic_agent import (
    MultiHeuristicPolicy,
    HEURISTIC_VERSION,
)  # noqa: F401


def make_opponent(name: str = "heuristic-v1") -> "MultiHeuristicPolicy":
    """Return an opponent policy by name.

    Parameters
    ----------
    name:
        Identifier for the opponent policy.  Currently only
        ``"heuristic-v1"`` is supported.

    Returns
    -------
    MultiHeuristicPolicy
        An instance of the frozen rule-based heuristic.

    Raises
    ------
    ValueError
        If *name* is not a known opponent.

    Example
    -------
    >>> from aircombat_bench.agents import make_opponent
    >>> opp = make_opponent("heuristic-v1")
    """
    if name == HEURISTIC_VERSION:
        return MultiHeuristicPolicy()
    raise ValueError(
        f"Unknown opponent '{name}'. Available: ['{HEURISTIC_VERSION}']"
    )
