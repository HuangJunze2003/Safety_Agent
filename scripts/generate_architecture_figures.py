#!/usr/bin/env python3
"""Generate architecture/sequence/deployment diagrams as PNG files."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"


def _box(ax, x, y, w, h, text, fc="#EAF2FF"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2,
        edgecolor="#355070",
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)


def _arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.2, color="#2F3E46"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.02, label, ha="center", va="bottom", fontsize=9, color="#1D3557")


def draw_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Safety Agent System Architecture", fontsize=14, weight="bold")

    _box(ax, 0.04, 0.72, 0.16, 0.14, "User / Inspector")
    _box(ax, 0.24, 0.72, 0.2, 0.14, "Gradio WebUI\nscripts/run_agent_webui.py")
    _box(ax, 0.48, 0.72, 0.24, 0.14, "SafetyProductionAgent.analyze()\nsrc/agent/workflow.py")
    _box(ax, 0.76, 0.72, 0.2, 0.14, "Qwen OpenAI API\n:8000/v1")

    _box(ax, 0.10, 0.42, 0.24, 0.14, "CLIPCaseEngine.search_similar_cases()\n(only non-legal + image)", fc="#FFF3E6")
    _box(ax, 0.38, 0.42, 0.24, 0.14, "LegalClauseRetriever.search()\n(skip for greeting)", fc="#FFF3E6")
    _box(ax, 0.66, 0.42, 0.28, 0.14, "Intent Router in Agent\nLLM classify + regex fallback", fc="#FFF3E6")

    _box(ax, 0.06, 0.14, 0.24, 0.14, "Image Store\n data/data_processed/images", fc="#F1FAEE")
    _box(ax, 0.34, 0.14, 0.24, 0.14, "Case Metadata\n cases_metadata.json", fc="#F1FAEE")
    _box(ax, 0.62, 0.14, 0.32, 0.14, "Vector DBs\n chroma_clip / chroma_legal", fc="#F1FAEE")

    _arrow(ax, 0.20, 0.79, 0.24, 0.79, "question/image")
    _arrow(ax, 0.44, 0.79, 0.48, 0.79, "analyze()")
    _arrow(ax, 0.60, 0.72, 0.80, 0.72, "intent classify + answer")
    _arrow(ax, 0.80, 0.76, 0.60, 0.76, "intent/answer")
    _arrow(ax, 0.52, 0.72, 0.78, 0.56, "intent detect")
    _arrow(ax, 0.56, 0.72, 0.50, 0.56, "legal retrieval")
    _arrow(ax, 0.54, 0.72, 0.22, 0.56, "case retrieval (conditional)")
    _arrow(ax, 0.22, 0.42, 0.18, 0.28, "read image")
    _arrow(ax, 0.22, 0.42, 0.44, 0.28, "read metadata")
    _arrow(ax, 0.50, 0.42, 0.78, 0.28, "query legal vector DB")
    _arrow(ax, 0.22, 0.42, 0.78, 0.28, "query clip vector DB")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_sequence(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Agent Runtime Sequence (Intent-Routed)", fontsize=14, weight="bold")

    lanes = {
        "User": 0.08,
        "WebUI": 0.26,
        "Agent": 0.46,
        "Retrievers": 0.68,
        "LLM": 0.88,
    }
    for name, x in lanes.items():
        ax.text(x, 0.95, name, ha="center", va="center", fontsize=10, weight="bold")
        ax.plot([x, x], [0.08, 0.92], linestyle="--", linewidth=1, color="#9AA0A6")

    steps = [
        ("User", "WebUI", 0.88, "submit question (+ optional image)"),
        ("WebUI", "Agent", 0.82, "call analyze()"),
        ("Agent", "LLM", 0.76, "intent classify (LLM-first)"),
        ("LLM", "Agent", 0.70, "intent label"),
        ("Agent", "Retrievers", 0.64, "legal retrieval (if not greeting)"),
        ("Agent", "Retrievers", 0.58, "case retrieval (if hazard + image)"),
        ("Retrievers", "Agent", 0.52, "retrieved laws/cases"),
        ("Agent", "LLM", 0.46, "compose prompt + ask final answer"),
        ("LLM", "Agent", 0.40, "final text/JSON"),
        ("Agent", "WebUI", 0.34, "intent + sources + structured_output"),
        ("WebUI", "User", 0.28, "render final response"),
    ]
    for src, dst, y, label in steps:
        _arrow(ax, lanes[src], y, lanes[dst], y, label)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_deployment(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Deployment Topology", fontsize=14, weight="bold")

    _box(ax, 0.03, 0.64, 0.20, 0.20, "Client Browser\n(Gradio UI)\n:7860")
    _box(ax, 0.28, 0.64, 0.34, 0.20, "Application Host\nAgent + Retrievers\nConda env llama")
    _box(ax, 0.66, 0.64, 0.30, 0.20, "Model Serving Host\nOpenAI-compatible API\n:8000")

    _box(ax, 0.06, 0.22, 0.28, 0.22, "Ingestion Service (offline/maintenance)\nFastAPI :8001", fc="#FFF3E6")
    _box(ax, 0.40, 0.22, 0.24, 0.22, "Persistent Storage\ncases_metadata + images", fc="#F1FAEE")
    _box(ax, 0.72, 0.22, 0.22, 0.22, "Chroma Vector DB\nclip + legal", fc="#F1FAEE")

    _arrow(ax, 0.23, 0.74, 0.28, 0.74, "HTTP")
    _arrow(ax, 0.62, 0.74, 0.66, 0.74, "chat completion")
    _arrow(ax, 0.46, 0.64, 0.52, 0.44, "read metadata/images")
    _arrow(ax, 0.56, 0.64, 0.83, 0.44, "query vectors")
    _arrow(ax, 0.34, 0.34, 0.40, 0.34, "write metadata/images")
    _arrow(ax, 0.64, 0.34, 0.72, 0.34, "re-index vectors")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    draw_architecture(FIG_DIR / "architecture.png")
    draw_sequence(FIG_DIR / "sequence.png")
    draw_deployment(FIG_DIR / "deployment.png")
    print("Generated figures:")
    print(f"- {FIG_DIR / 'architecture.png'}")
    print(f"- {FIG_DIR / 'sequence.png'}")
    print(f"- {FIG_DIR / 'deployment.png'}")


if __name__ == "__main__":
    main()
