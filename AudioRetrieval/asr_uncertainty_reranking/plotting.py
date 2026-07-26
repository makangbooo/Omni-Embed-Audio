"""Non-overwriting result plots; matplotlib is imported only when used."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence


def _destination(path: Path | str) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite plot: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def plot_noise_curve(
    metrics_by_method: Mapping[str, Mapping[str, float]],
    *,
    metric: str,
    output: Path | str,
    condition_order: Sequence[str] = ("clean", "snr_20", "snr_10", "snr_0"),
) -> None:
    destination = _destination(output)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    for method, conditions in sorted(metrics_by_method.items()):
        missing = [condition for condition in condition_order if condition not in conditions]
        if missing:
            raise KeyError(f"{method} missing conditions: {missing}")
        axis.plot(
            range(len(condition_order)),
            [conditions[condition] for condition in condition_order],
            marker="o",
            label=method,
        )
    axis.set_xticks(range(len(condition_order)), condition_order)
    axis.set_ylabel(metric)
    axis.set_xlabel("Acoustic condition")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_gate_distribution(
    gates_by_condition: Mapping[str, Sequence[float]],
    *,
    output: Path | str,
) -> None:
    destination = _destination(output)
    import matplotlib.pyplot as plt

    conditions = list(gates_by_condition)
    if not conditions or any(not gates_by_condition[value] for value in conditions):
        raise ValueError("every condition must contain gate values")
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.violinplot(
        [gates_by_condition[condition] for condition in conditions],
        showmeans=True,
        showextrema=True,
    )
    axis.set_xticks(range(1, len(conditions) + 1), conditions)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("ASR gate weight")
    axis.set_xlabel("Acoustic condition")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)
