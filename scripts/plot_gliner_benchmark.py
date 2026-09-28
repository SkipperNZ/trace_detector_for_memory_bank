"""Render an aggregate-only benchmark figure without loading models or input texts."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("docs/experiments/gliner25-decide-2026-09-28-metrics.json"),
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    result = json.loads(args.metrics.read_text(encoding="utf-8"))["evaluation"]
    models = [
        {"name": "GLiNER2.5-Decide · без дообучения", "test": result["test"]},
        *result["baselines"],
    ]
    models.sort(key=lambda r: r["test"]["mean_macro_f1"], reverse=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig = plt.figure(figsize=(11, 9.5), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    ax = fig.add_subplot(grid[0, :])
    bars = ax.barh(
        [r["name"] for r in models],
        [r["test"]["mean_macro_f1"] for r in models],
        color=["#be4b35" if r["name"].startswith("GLiNER") else "#327d9d" for r in models],
    )
    ax.bar_label(bars, fmt="%.3f", padding=6)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.75)
    ax.set_xlabel("Mean macro-F1 · три класса, две задачи")
    ax.set_title("Один test: 1 348 сообщений с метками Qwen", loc="left", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    for col, (task, title) in enumerate(
        [("dissatisfaction", "Недовольство агентом"), ("correction", "Исправление агента")]
    ):
        ax = fig.add_subplot(grid[1, col])
        matrix = np.asarray(result["test"]["tasks"][task]["confusion_matrix"])
        fractions = matrix / matrix.sum(axis=1, keepdims=True)
        ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1)
        for i, j in np.ndindex(matrix.shape):
            ax.text(
                j,
                i,
                f"{matrix[i, j]}\n{fractions[i, j]:.1%}",
                ha="center",
                va="center",
                color="white" if fractions[i, j] > 0.55 else "#172733",
            )
        ax.set_xticks(range(3), ["no", "yes", "unclear"])
        ax.set_yticks(range(3), ["no", "yes", "unclear"])
        ax.set_xlabel("Предсказание GLiNER")
        ax.set_ylabel("Метка Qwen")
        ax.set_title(title, pad=12)
    fig.suptitle("GLiNER2.5-Decide · проверка на feedback-v1", fontsize=17)
    fig.supxlabel(
        "Матрицы: число сообщений и доля внутри строки. Метки Qwen — не человеческий эталон.\n"
        "GLiNER без обучения весов; остальные модели обучены на train. Схема GLiNER выбрана на calibration.",
        fontsize=10,
    )
    output = args.out or args.metrics.with_name(args.metrics.stem.removesuffix("-metrics") + ".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
