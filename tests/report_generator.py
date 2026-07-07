"""Генератор отчёта по датасету."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src.fast_parser import fast_parse
from src.deep_interpreter import deep_interpret
from src.fusion import apply_fusion_layer

def analyze_dataset(dataset_dir="tests/dataset"):
    dataset_path = Path(dataset_dir)
    results = []
    for txt_file in dataset_path.glob("*.txt"):
        author = txt_file.stem
        text = txt_file.read_text(encoding="utf-8").strip()
        sentences = text.split("\n")
        for sentence in sentences:
            if not sentence.strip():
                continue
            try:
                syntax = fast_parse(sentence, use_cache=True)
                heidegger = deep_interpret(sentence, syntax, use_cache=True)
                fusion = apply_fusion_layer(syntax, heidegger)
                results.append({
                    "author": author,
                    "sentence": sentence,
                    "final_index": fusion["final_index"],
                    "health_status": fusion["health_status"],
                    "dasein_mode": heidegger.dasein_mode,
                    "hold_break_risk": heidegger.hold_break_risk
                })
            except Exception as e:
                results.append({"author": author, "sentence": sentence, "error": str(e)})
    return results

def generate_report(results, output_file="tests/report.json"):
    Path(output_file).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report saved to: {output_file}")

def plot_results(results, output_file="tests/plot.png"):
    authors = {}
    for r in results:
        if "error" in r:
            continue
        author = r["author"]
        if author not in authors:
            authors[author] = []
        authors[author].append(r["final_index"])
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(authors))
    means = [np.mean(v) for v in authors.values()]
    stds = [np.std(v) for v in authors.values()]
    ax.bar(positions, means, yerr=stds, capsize=5, alpha=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(authors.keys(), rotation=45, ha="right")
    ax.set_ylabel("Final Index")
    ax.set_title("GrammaLang Hybrid: Index of Will by Author")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to: {output_file}")

if __name__ == "__main__":
    res = analyze_dataset()
    generate_report(res)
    plot_results(res)
    print("Done!")
