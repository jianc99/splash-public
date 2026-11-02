"""The interval rule of the 1.6.0 benchmark matrix, stdlib only.

A gate compares a candidate's runs with a reference's on the log of a
per-run metric. Per run the noise is the machine's measured per-run spread of
that metric between runs of one tree in one order (sigma, from recorded
same-tree pairs), so the difference of the two means has a one-sided 90 %
interval of half-width t(0.9, df) * sigma * sqrt(1/n_cand + 1/n_ref). Orders
are compared within themselves and pooled as the mean of their differences.
A tolerance turns the interval into a PASS, and equality (or the loss a
gate allows) into a FAIL; noise only decides how many runs resolve it, never
the tolerance.
"""

from __future__ import annotations

import math
import statistics

PASS, FAIL, INCONCLUSIVE = "pass", "fail", "inconclusive"
CONFIDENCE = 0.9


def _continued_fraction(a: float, b: float, x: float) -> float:
    tiny, epsilon = 1e-300, 3e-16
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        for numerator in (
            m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
            -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1)),
        ):
            d = 1.0 + numerator * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + numerator / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < epsilon:
            break
    return h


def incomplete_beta(a: float, b: float, x: float) -> float:
    """The regularized incomplete beta function I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1 - x)
    )
    if x < (a + 1) / (a + b + 2):
        return front * _continued_fraction(a, b, x) / a
    return 1.0 - front * _continued_fraction(b, a, 1 - x) / b


def t_cdf(t: float, df: float) -> float:
    tail = 0.5 * incomplete_beta(df / 2, 0.5, df / (df + t * t))
    return 1.0 - tail if t >= 0 else tail


def t_quantile(p: float, df: float) -> float:
    """The p quantile of Student's t with df degrees of freedom, p > 0.5."""
    low, high = 0.0, 1000.0
    for _ in range(200):
        middle = (low + high) / 2
        if t_cdf(middle, df) < p:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def pooled_sigma(groups: list[list[float]]) -> tuple[float | None, int]:
    """The per-run standard deviation of log values within groups of one
    tree in one order, pooled, and its degrees of freedom."""
    squares, df = 0.0, 0
    for group in groups:
        logs = [math.log(v) for v in group if v and v > 0]
        if len(logs) > 1:
            mean = statistics.fmean(logs)
            squares += sum((v - mean) ** 2 for v in logs)
            df += len(logs) - 1
    return (math.sqrt(squares / df), df) if df else (None, 0)


def interval(by_order: dict, sigma: float, df: int) -> dict:
    """The log ratio candidate/reference per order and pooled, with each
    one-sided 90 % half-width. by_order: {order: (candidate values,
    reference values)}; orders lacking either side are left out."""
    t = t_quantile(CONFIDENCE, df)
    orders, differences, inverse = {}, [], []
    for order, (candidate, reference) in sorted(by_order.items()):
        candidate = [v for v in candidate if v and v > 0]
        reference = [v for v in reference if v and v > 0]
        if not candidate or not reference:
            continue
        difference = statistics.fmean(map(math.log, candidate)) - statistics.fmean(
            map(math.log, reference)
        )
        weight = 1 / len(candidate) + 1 / len(reference)
        orders[order] = {
            "log_ratio": difference,
            "half_width": t * sigma * math.sqrt(weight),
            "n_candidate": len(candidate),
            "n_reference": len(reference),
        }
        differences.append(difference)
        inverse.append(weight)
    if not differences:
        return {"orders": orders, "pooled": None, "sigma": sigma, "df": df, "t": t}
    pooled = {
        "log_ratio": statistics.fmean(differences),
        "half_width": t * sigma * math.sqrt(sum(inverse)) / len(differences),
    }
    return {"orders": orders, "pooled": pooled, "sigma": sigma, "df": df, "t": t}


def verdict(
    estimate: dict | None,
    higher_is_better: bool,
    tolerance: float,
    fail_tolerance: float = 0.0,
) -> str:
    """PASS when the bound on the good side clears the tolerance, else FAIL
    when the bound on the bad side is past fail_tolerance (equality unless a
    gate allows a loss, DESIGN-1.6.0 §8.3, amend/V.md §5), else
    INCONCLUSIVE."""
    if estimate is None:
        return INCONCLUSIVE
    low = estimate["log_ratio"] - estimate["half_width"]
    high = estimate["log_ratio"] + estimate["half_width"]
    if higher_is_better:
        passing, failing = math.log(1 - tolerance), math.log(1 - fail_tolerance)
        return PASS if low >= passing else FAIL if high < failing else INCONCLUSIVE
    passing, failing = math.log(1 + tolerance), math.log(1 + fail_tolerance)
    return PASS if high <= passing else FAIL if low > failing else INCONCLUSIVE


def clearly_better(estimate: dict | None, higher_is_better: bool) -> bool:
    """The whole interval lies on the good side of equality."""
    if estimate is None:
        return False
    if higher_is_better:
        return estimate["log_ratio"] - estimate["half_width"] > 0
    return estimate["log_ratio"] + estimate["half_width"] < 0


def ratio_text(estimate: dict | None) -> str:
    if estimate is None:
        return "n/a"
    low = math.exp(estimate["log_ratio"] - estimate["half_width"])
    high = math.exp(estimate["log_ratio"] + estimate["half_width"])
    return f"{math.exp(estimate['log_ratio']):.3f} [{low:.3f}, {high:.3f}]"


def geometric_mean(values: list[float]) -> float | None:
    values = [v for v in values if v and v > 0]
    return math.exp(statistics.fmean(map(math.log, values))) if values else None
