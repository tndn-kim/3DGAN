"""
visualization.py
----------------
증강 전/후 분류 성능 비교 시각화 모듈.

출력 그래프:
  1. metric_comparison.png  : 모델별 Accuracy / F1 / FNR 증강 전후 비교 막대 그래프
  2. confusion_matrices.png : 모델별 증강 전후 Confusion Matrix 히트맵
  3. summary_radar.png      : 모델별 종합 성능 레이더 차트
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns


# ────────────────────────────────────────────────
# 스타일 설정
# ────────────────────────────────────────────────

COLOR_BEFORE = "#5B8DB8"   # 증강 전 (파랑)
COLOR_AFTER  = "#E07B54"   # 증강 후 (주황)
MODEL_LABELS = {"rf": "Random\nForest", "xgb": "XGBoost", "bigru": "BiGRU"}
METRIC_LABELS = {
    "accuracy" : "Accuracy",
    "f1_macro" : "F1-Score\n(Macro)",
    "micro_FNR": "FNR\n(미탐, ↓낮을수록 좋음)",
}

def _set_style():
    plt.rcParams.update({
        "font.family"      : "DejaVu Sans",
        "axes.spines.top"  : False,
        "axes.spines.right": False,
        "axes.grid"        : True,
        "grid.alpha"       : 0.3,
        "grid.linestyle"   : "--",
    })


# ────────────────────────────────────────────────
# 그래프 1: 지표별 막대 비교
# ────────────────────────────────────────────────

def _plot_metric_comparison(results_before: dict,
                             results_after: dict,
                             save_path: str):
    """
    Accuracy / F1 / FNR 세 지표를 모델별로 증강 전후 비교.
    각 지표마다 서브플롯 1개, 총 3개.
    """
    _set_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("증강 전/후 분류 성능 비교", fontsize=16, fontweight="bold", y=1.02)

    model_keys   = [k for k in MODEL_LABELS if k in results_before]
    x            = np.arange(len(model_keys))
    bar_w        = 0.35

    metrics = ["accuracy", "f1_macro", "micro_FNR"]

    for ax, metric in zip(axes, metrics):
        vals_before = [results_before[m]["metrics"][metric] for m in model_keys]
        vals_after  = [results_after[m]["metrics"][metric]  for m in model_keys]

        bars_b = ax.bar(x - bar_w/2, vals_before, bar_w,
                        label="증강 전", color=COLOR_BEFORE,
                        alpha=0.85, edgecolor="white", linewidth=0.8)
        bars_a = ax.bar(x + bar_w/2, vals_after,  bar_w,
                        label="증강 후", color=COLOR_AFTER,
                        alpha=0.85, edgecolor="white", linewidth=0.8)

        # 값 라벨
        for bar in bars_b:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=8.5, color=COLOR_BEFORE, fontweight="bold")
        for bar in bars_a:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=8.5, color=COLOR_AFTER, fontweight="bold")

        # 증강 후 변화량 화살표
        for i, (vb, va) in enumerate(zip(vals_before, vals_after)):
            delta = va - vb
            sign  = "▲" if delta > 0 else "▼"
            # FNR은 낮을수록 좋으므로 색 반전
            good  = (delta > 0) if metric != "micro_FNR" else (delta < 0)
            color = "#2ECC71" if good else "#E74C3C"
            ax.text(x[i], max(vb, va) + 0.025,
                    f"{sign}{abs(delta):.3f}",
                    ha="center", fontsize=8, color=color, fontweight="bold")

        ax.set_title(METRIC_LABELS[metric], fontsize=12, fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in model_keys], fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score", fontsize=9)
        ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ 저장: {save_path}")


# ────────────────────────────────────────────────
# 그래프 2: Confusion Matrix 히트맵
# ────────────────────────────────────────────────

def _plot_confusion_matrices(results_before: dict,
                              results_after: dict,
                              save_path: str):
    """
    모델별 (행) × 증강 전/후 (열) Confusion Matrix 히트맵.
    """
    _set_style()
    model_keys = [k for k in MODEL_LABELS if k in results_before]
    n_models   = len(model_keys)

    fig, axes = plt.subplots(n_models, 2,
                              figsize=(10, 4.5 * n_models))
    fig.suptitle("Confusion Matrix 비교 (증강 전 / 증강 후)",
                 fontsize=15, fontweight="bold", y=1.01)

    for row, mkey in enumerate(model_keys):
        for col, (tag, results) in enumerate(
            [("증강 전", results_before), ("증강 후", results_after)]
        ):
            ax   = axes[row][col] if n_models > 1 else axes[col]
            conf = results[mkey]["metrics"]["conf_matrix"]
            n    = conf.shape[0]

            # 정규화 (행 합 = 1) → 비율로 색상, 원본 count는 텍스트
            conf_norm = conf.astype(float) / conf.sum(axis=1, keepdims=True)

            sns.heatmap(
                conf_norm, ax=ax,
                annot=False, fmt=".2f",
                cmap="Blues" if col == 0 else "Oranges",
                vmin=0, vmax=1,
                linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.8},
            )

            # 셀 텍스트: count + 비율
            for i in range(n):
                for j in range(n):
                    count = conf[i, j]
                    ratio = conf_norm[i, j]
                    color = "white" if ratio > 0.6 else "black"
                    ax.text(j + 0.5, i + 0.5,
                            f"{count:,}\n({ratio:.1%})",
                            ha="center", va="center",
                            fontsize=9, color=color, fontweight="bold")

            acc = results[mkey]["metrics"]["accuracy"]
            f1  = results[mkey]["metrics"]["f1_macro"]
            fnr = results[mkey]["metrics"]["micro_FNR"]

            ax.set_title(
                f"{MODEL_LABELS[mkey].replace(chr(10), ' ')} │ {tag}\n"
                f"Acc {acc:.3f}  F1 {f1:.3f}  FNR {fnr:.3f}",
                fontsize=10, fontweight="bold",
            )
            ax.set_xlabel("예측 레이블", fontsize=9)
            ax.set_ylabel("실제 레이블", fontsize=9)
            tick_labels = [f"Class {i}" for i in range(n)]
            ax.set_xticklabels(tick_labels, fontsize=8)
            ax.set_yticklabels(tick_labels, fontsize=8, rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ 저장: {save_path}")


# ────────────────────────────────────────────────
# 그래프 3: 레이더 차트 (종합 비교)
# ────────────────────────────────────────────────

def _plot_radar(results_before: dict,
                results_after: dict,
                save_path: str):
    """
    모델별 레이더 차트로 증강 전후 종합 비교.
    FNR은 (1 - FNR) 로 변환해 높을수록 좋은 방향으로 통일.
    """
    _set_style()
    model_keys  = [k for k in MODEL_LABELS if k in results_before]
    metric_keys = ["accuracy", "f1_macro", "micro_FNR"]
    radar_labels = ["Accuracy", "F1-Score", "1 - FNR\n(미탐 역전)"]
    n_metrics    = len(metric_keys)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]   # 폐곡선

    fig, axes = plt.subplots(1, len(model_keys),
                              figsize=(6 * len(model_keys), 6),
                              subplot_kw=dict(polar=True))
    if len(model_keys) == 1:
        axes = [axes]

    fig.suptitle("증강 전/후 종합 성능 레이더 차트",
                 fontsize=15, fontweight="bold")

    for ax, mkey in zip(axes, model_keys):
        def _get_vals(results):
            vals = []
            for mk in metric_keys:
                v = results[mkey]["metrics"][mk]
                vals.append(1 - v if mk == "micro_FNR" else v)
            return vals + [vals[0]]

        vb = _get_vals(results_before)
        va = _get_vals(results_after)

        ax.plot(angles, vb, color=COLOR_BEFORE, linewidth=2, label="증강 전")
        ax.fill(angles, vb, color=COLOR_BEFORE, alpha=0.2)
        ax.plot(angles, va, color=COLOR_AFTER,  linewidth=2, label="증강 후")
        ax.fill(angles, va, color=COLOR_AFTER,  alpha=0.2)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_labels, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.50, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"],
                           fontsize=7, color="grey")
        ax.set_title(MODEL_LABELS[mkey].replace("\n", " "),
                     fontsize=12, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
        ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ 저장: {save_path}")


# ────────────────────────────────────────────────
# 메인 시각화 함수
# ────────────────────────────────────────────────

def visualize(config: dict,
              results_before: dict,
              results_after: dict) -> None:
    """
    증강 전/후 분류 결과를 3종 그래프로 시각화.

    Args:
        config:
            - save_dir : 그래프 저장 경로
        results_before : classify() 반환값 (증강 전 데이터로 학습)
        results_after  : classify() 반환값 (증강 후 데이터로 학습)

    pipeline.py 사용 예시:
        from visualization import visualize
        visualize(config, results_before, results_after)
    """
    save_dir = config.get("save_dir", "./results")
    os.makedirs(save_dir, exist_ok=True)

    print("\n[시각화] 그래프 생성 시작")

    _plot_metric_comparison(
        results_before, results_after,
        save_path=os.path.join(save_dir, "metric_comparison.png"),
    )
    _plot_confusion_matrices(
        results_before, results_after,
        save_path=os.path.join(save_dir, "confusion_matrices.png"),
    )
    _plot_radar(
        results_before, results_after,
        save_path=os.path.join(save_dir, "summary_radar.png"),
    )

    print(f"\n✅ 시각화 완료 → {save_dir}/")
    print("   metric_comparison.png  : 지표별 막대 비교")
    print("   confusion_matrices.png : Confusion Matrix 히트맵")
    print("   summary_radar.png      : 레이더 차트")