#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import requests


@dataclass
class ModelEndpoint:
    name: str
    base_url: str
    model: str
    api_key: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测原始与微调后 Qwen 3.0-VL 的性能差异")
    parser.add_argument("--eval-data", type=Path, required=True, help="评测集 JSONL 路径（建议 50 条）")
    parser.add_argument("--base-url", type=str, required=True, help="原始模型 OpenAI-compatible API 地址")
    parser.add_argument("--base-model", type=str, required=True, help="原始模型名称")
    parser.add_argument("--ft-url", type=str, required=True, help="微调模型 OpenAI-compatible API 地址")
    parser.add_argument("--ft-model", type=str, required=True, help="微调模型名称")
    parser.add_argument("--api-key", type=str, default="", help="可选 API Key")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/data_processed/eval_reports"),
        help="评测结果输出目录",
    )
    return parser.parse_args()


def load_eval_samples(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"评测数据不存在: {path}")

    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_no} 行 JSON 解析失败") from exc
            samples.append(sample)
    return samples


def call_model(endpoint: ModelEndpoint, messages: list[dict[str, Any]]) -> tuple[str, float]:
    url = endpoint.base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if endpoint.api_key:
        headers["Authorization"] = f"Bearer {endpoint.api_key}"

    payload = {
        "model": endpoint.model,
        "messages": messages,
        "temperature": 0.1,
    }

    start = time.perf_counter()
    response = requests.post(url, headers=headers, json=payload, timeout=180)
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    body = response.json()

    content = body["choices"][0]["message"].get("content", "")
    return str(content), elapsed


def extract_risk_level(text: str) -> str:
    block_match = re.search(r"【风险等级】\s*(.*?)\s*(?:【|$)", text, flags=re.S)
    candidate = block_match.group(1).strip() if block_match else text
    for label in ["高风险", "中风险", "一般风险"]:
        if label in candidate:
            return label
    return "未知"


def law_citation_hit(response_text: str, reference_law: str) -> int:
    if reference_law:
        return int(reference_law in response_text)
    has_article = bool(re.search(r"第.{0,6}条", response_text))
    has_law_name = any(name in response_text for name in ["安全生产法", "四川省有限空间作业管理规定"])
    return int(has_article and has_law_name)


def risk_consistency(response_text: str, reference_risk: str) -> int:
    pred = extract_risk_level(response_text)
    if not reference_risk:
        return int(pred != "未知")
    return int(pred == reference_risk)


def terminology_score(response_text: str) -> float:
    terms = ["有限空间", "风险分级", "作业许可", "个人防护", "通风", "监护", "警示标识", "整改闭环"]
    hit = sum(1 for term in terms if term in response_text)
    return hit / len(terms)


def logic_score(response_text: str) -> float:
    required_sections = ["[视觉诊断]", "[合规对标]", "[整改方案]"]
    return sum(1 for sec in required_sections if sec in response_text) / len(required_sections)


def evaluate_model(endpoint: ModelEndpoint, samples: list[dict[str, Any]]) -> dict[str, Any]:
    law_hits = 0
    risk_hits = 0
    total_time = 0.0
    term_scores: list[float] = []
    logic_scores: list[float] = []
    details: list[dict[str, Any]] = []

    for sample in samples:
        messages = sample.get("messages", [])
        if not isinstance(messages, list):
            continue

        response_text, elapsed = call_model(endpoint, messages)
        total_time += elapsed

        reference_law = str(sample.get("reference_law", "")).strip()
        reference_risk = str(sample.get("reference_risk", "")).strip()

        law_hit = law_citation_hit(response_text, reference_law)
        risk_hit = risk_consistency(response_text, reference_risk)
        t_score = terminology_score(response_text)
        l_score = logic_score(response_text)

        law_hits += law_hit
        risk_hits += risk_hit
        term_scores.append(t_score)
        logic_scores.append(l_score)

        details.append(
            {
                "id": sample.get("id", ""),
                "response_time_sec": round(elapsed, 4),
                "law_hit": law_hit,
                "risk_hit": risk_hit,
                "terminology_score": round(t_score, 4),
                "logic_score": round(l_score, 4),
            }
        )

    n = max(len(details), 1)
    return {
        "model_name": endpoint.name,
        "num_samples": len(details),
        "law_citation_accuracy": law_hits / n,
        "risk_level_consistency": risk_hits / n,
        "avg_response_time_sec": total_time / n,
        "terminology_accuracy": float(np.mean(term_scores) if term_scores else 0.0),
        "logical_coherence": float(np.mean(logic_scores) if logic_scores else 0.0),
        "details": details,
    }


def plot_radar(base_result: dict[str, Any], ft_result: dict[str, Any], output_path: Path) -> None:
    labels = ["专业术语准确性", "逻辑条理性"]
    base_values = [base_result["terminology_accuracy"], base_result["logical_coherence"]]
    ft_values = [ft_result["terminology_accuracy"], ft_result["logical_coherence"]]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    base_plot = base_values + base_values[:1]
    ft_plot = ft_values + ft_values[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.plot(angles, base_plot, linewidth=2, label="原始 Qwen 3.0")
    ax.fill(angles, base_plot, alpha=0.15)
    ax.plot(angles, ft_plot, linewidth=2, label="微调后 Qwen 3.0")
    ax.fill(angles, ft_plot, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("微调前后能力对比雷达图", fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    samples = load_eval_samples(args.eval_data)

    if len(samples) < 50:
        print(f"[WARN] 当前评测样本数为 {len(samples)}，建议使用 50 条样本以满足论文对比要求。")

    base_endpoint = ModelEndpoint(
        name="base_qwen3vl",
        base_url=args.base_url,
        model=args.base_model,
        api_key=args.api_key or None,
    )
    ft_endpoint = ModelEndpoint(
        name="finetuned_qwen3vl",
        base_url=args.ft_url,
        model=args.ft_model,
        api_key=args.api_key or None,
    )

    base_result = evaluate_model(base_endpoint, samples)
    ft_result = evaluate_model(ft_endpoint, samples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "comparison_report.json"
    radar_path = args.output_dir / "radar_improvement.png"

    report = {
        "base": base_result,
        "finetuned": ft_result,
        "delta": {
            "law_citation_accuracy": ft_result["law_citation_accuracy"] - base_result["law_citation_accuracy"],
            "risk_level_consistency": ft_result["risk_level_consistency"] - base_result["risk_level_consistency"],
            "avg_response_time_sec": ft_result["avg_response_time_sec"] - base_result["avg_response_time_sec"],
            "terminology_accuracy": ft_result["terminology_accuracy"] - base_result["terminology_accuracy"],
            "logical_coherence": ft_result["logical_coherence"] - base_result["logical_coherence"],
        },
    }

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_radar(base_result, ft_result, radar_path)

    print(json.dumps({
        "report": str(report_path),
        "radar": str(radar_path),
        "base_law_acc": round(base_result["law_citation_accuracy"], 4),
        "ft_law_acc": round(ft_result["law_citation_accuracy"], 4),
        "base_risk_consistency": round(base_result["risk_level_consistency"], 4),
        "ft_risk_consistency": round(ft_result["risk_level_consistency"], 4),
        "base_avg_latency_sec": round(base_result["avg_response_time_sec"], 4),
        "ft_avg_latency_sec": round(ft_result["avg_response_time_sec"], 4),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
