"""Генерация демо-данных кредитного скоринга с искусственно внесённым дрейфом.

Запуск:
    python scripts/generate_demo.py --out data/demo --seed 42

Создаёт reference.csv (эталон) и current_<сценарий>.csv для каждого сценария
из ``drift_guardian.demo.SCENARIOS``. Seed фиксирован — данные воспроизводимы.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from drift_guardian.demo import SCENARIOS, make_current, make_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/demo", help="каталог для CSV-файлов")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ref-rows", type=int, default=20_000)
    parser.add_argument("--cur-rows", type=int, default=5_000)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference = make_reference(args.ref_rows, args.seed)
    reference.to_csv(out_dir / "reference.csv", index=False)
    print(f"reference.csv: {len(reference)} строк")

    for name, description in SCENARIOS.items():
        batch = make_current(name, args.cur_rows, args.seed)
        path = out_dir / f"current_{name}.csv"
        batch.to_csv(path, index=False)
        print(f"{path.name}: {len(batch)} строк — {description}")
    print(f"\nГотово. Файлы в {out_dir.resolve()}")


if __name__ == "__main__":
    main()
