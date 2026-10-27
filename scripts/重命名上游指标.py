"""只更改上游展示表头，不改变历史实验数值。"""
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/upstream"
FILES = ("main_summary.csv", "vocabulary_sweep.csv", "three_metrics.csv")
NAMES = {
    "compression": "tokenizer compression",
    "compression_ratio": "tokenizer compression",
    "cmd": "operation",
    "cmd_pct": "operation",
    "num": "numbers",
    "num_pct": "numbers",
}


def main():
    for name in FILES:
        path = REPORT / name
        rename(path)
    rename(ROOT / "reports/vocab_usage/three_metrics.csv")


def rename(path):
    name = path.name
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        rows = list(reader)
    if not columns:
        raise ValueError(f"空表：{path}")
    updated = [NAMES.get(column, column) for column in columns]
    if len(set(updated)) != len(columns):
        raise ValueError(f"列名冲突：{path}")
    if updated == columns:
        print(name, "表头已更新")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=updated)
        writer.writeheader()
        for row in rows:
            writer.writerow({NAMES.get(column, column): value for column, value in row.items()})
    print(name, len(rows), "条")


if __name__ == "__main__":
    main()
