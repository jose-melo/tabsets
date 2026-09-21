"""Set-valued reliability for tabular classifiers, from cached probabilities.

Everything here is a function of four arrays that one run of a classifier produces:
the class probabilities and labels on a calibration split, and the same on a test
split. From those, at any level and for any conformity score, come the prediction
sets, the coverage they deliver, how wide they are, how often they are empty, and the
two factors the article separates: how often a model commits to at least one label,
and how well it covers the truth where it does.

    >>> import tabsets
    >>> r = tabsets.evaluate(p_cal, y_cal, p_test, y_test, alpha=0.10, score="lac")
    >>> r["commitment"], r["sscs_plus"], r["guarantee"]

Sets built from labelled calibration data reproduce MAPIE's split-conformal classifier
exactly on multiclass targets, which is what makes the released numbers checkable
against an independent implementation. On binary targets MAPIE declines the adaptive,
regularised and top-k scores; those sets are this package's own and say so in
``implementation``.

:mod:`tabsets.labelfree` builds sets without any labels at all. Those carry no coverage
guarantee and report ``guarantee=None``.
"""
from __future__ import annotations

__version__ = "0.1.0"

_SUBMODULES = {"cache", "sets", "metrics", "labelfree", "stats", "blocks", "cli"}


def __getattr__(name):
    if name in _SUBMODULES:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(name)


def __dir__():
    return sorted(_SUBMODULES | {"evaluate", "labelfree_threshold", "labelfree_sets", "__version__"})


def evaluate(p_cal, y_cal, p_test, y_test, alpha=0.10, score="lac", seed=0, **kw) -> dict:
    """One run, scored: the sets at ``alpha`` and everything the article reports of them.

    ``score`` is one of ``lac``, ``aps``, ``raps`` or ``top_k``. ``seed`` drives the
    tie-breaking the randomised scores need, and should be the seed the run itself was
    fitted with rather than a fixed one, or every model of a split shares a draw none of
    them made.
    """
    from . import metrics as _m
    from . import sets as _s

    r = _s.build_sets(score, p_cal, y_cal, p_test, alpha=alpha, seed=seed, **kw)
    d = _m.decomposition(y_test, r.sets)
    d.update(
        score=r.score, alpha=r.alpha, qhat=r.qhat, n_cal=r.n_cal,
        implementation=r.implementation, guarantee=r.guarantee,
        coverage=_m.coverage(y_test, r.sets),
        width=_m.mean_width(r.sets),
        singleton_rate=_m.singleton_rate(r.sets),
    )
    return d


def labelfree_threshold(p, alpha: float = 0.10, variant: str = "block") -> float:
    """See :func:`tabsets.labelfree.labelfree_threshold`."""
    from .labelfree import labelfree_threshold as _f
    return _f(p, alpha, variant=variant)


def labelfree_sets(p_test, alpha: float = 0.10, p_solve=None, variant: str = "block"):
    """See :func:`tabsets.labelfree.labelfree_sets`."""
    from .labelfree import labelfree_sets as _f
    return _f(p_test, alpha=alpha, p_solve=p_solve, variant=variant)


__all__ = ["evaluate", "labelfree_threshold", "labelfree_sets", "__version__", *sorted(_SUBMODULES)]
