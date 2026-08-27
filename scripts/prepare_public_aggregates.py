from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from rig_hazard.public_recurrence import aggregate_ecommerce, aggregate_us_accidents


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare compact public recurrent-event aggregates")
    parser.add_argument(
        "--config", default="configs/rig_hazard_alert_governance.json"
    )
    parser.add_argument(
        "--output-root",
        default="results/alert_governance/public_recurrence_benchmarks/prepared",
    )
    args = parser.parse_args()
    config = json.loads((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))
    settings = config["public_benchmarks"]
    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    ecommerce = settings["ecommerce"]
    ecommerce_source = PROJECT_ROOT / "data/public_benchmarks/2019-Nov.csv"
    ecommerce_output = aggregate_ecommerce(
        ecommerce_source,
        output_root / "ecommerce_hourly.parquet",
        int(ecommerce["maximum_entities"]),
    )
    print(f"Ecommerce aggregate ready: {ecommerce_output}", flush=True)

    accidents = settings["us_accidents"]
    accidents_source = PROJECT_ROOT / "data/public_benchmarks/US_Accidents_March23.csv"
    accidents_output = aggregate_us_accidents(
        accidents_source,
        output_root / "us_accidents_daily.parquet",
        int(accidents["maximum_entities"]),
        float(accidents["grid_degrees"]),
    )
    print(f"US accidents aggregate ready: {accidents_output}", flush=True)

    import duckdb

    outputs = {}
    for name, source, output in (
        ("ecommerce", ecommerce_source, ecommerce_output),
        ("us_accidents", accidents_source, accidents_output),
    ):
        escaped = str(output.resolve()).replace("\\", "/").replace("'", "''")
        row = duckdb.sql(
            f"SELECT count(*) AS rows, count(DISTINCT entity_id) AS entities, "
            f"sum(event_count) AS events FROM read_parquet('{escaped}')"
        ).fetchone()
        outputs[name] = {
            "source_name": source.name,
            "source_size_bytes": source.stat().st_size,
            "aggregate_name": output.name,
            "aggregate_size_bytes": output.stat().st_size,
            "rows": int(row[0]),
            "entities": int(row[1]),
            "events": int(row[2]),
        }
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "preprocessing": "DuckDB streaming aggregation from the two original public CSV files",
        "parameters": {
            "ecommerce_maximum_entities": int(ecommerce["maximum_entities"]),
            "accident_maximum_entities": int(accidents["maximum_entities"]),
            "accident_grid_degrees": float(accidents["grid_degrees"]),
        },
        "outputs": outputs,
        "scientific_role": "cross-domain recurrent-event benchmark, not external icing validation",
    }
    (output_root / "public_aggregate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
