# scripts/export_data_preview.py

from __future__ import annotations

from pathlib import Path
import pandas as pd

import config


def describe_parquet(path: Path, sample_rows: int = 200) -> tuple[dict, pd.DataFrame]:
    df = pd.read_parquet(path)

    info = {
        "file": path.name,
        "path": str(path),
        "rows": len(df),
        "columns": len(df.columns),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
    }

    schema_rows = []
    for col in df.columns:
        s = df[col]
        schema_rows.append({
            "file": path.name,
            "column": col,
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "missing": int(s.isna().sum()),
            "missing_pct": float(s.isna().mean()),
            "sample_value": None if s.dropna().empty else str(s.dropna().iloc[0]),
        })

    schema = pd.DataFrame(schema_rows)
    preview = df.head(sample_rows)

    return info, schema, preview


def main():
    processed_dir = Path(config.PathConfig.PROCESSED)
    output_dir = Path("docs/data_catalog")
    preview_dir = output_dir / "previews"

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(processed_dir.glob("*.parquet"))

    file_infos = []
    schemas = []

    for path in files:
        info, schema, preview = describe_parquet(path)
        file_infos.append(info)
        schemas.append(schema)

        preview_path = preview_dir / f"{path.stem}_preview.csv"
        preview.to_csv(preview_path, index=False, encoding="utf-8-sig")

    files_df = pd.DataFrame(file_infos)
    schema_df = pd.concat(schemas, ignore_index=True) if schemas else pd.DataFrame()

    files_df.to_csv(output_dir / "processed_files.csv", index=False, encoding="utf-8-sig")
    schema_df.to_csv(output_dir / "processed_schema.csv", index=False, encoding="utf-8-sig")

    print(f"Exported file list: {output_dir / 'processed_files.csv'}")
    print(f"Exported schema   : {output_dir / 'processed_schema.csv'}")
    print(f"Exported previews : {preview_dir}")


if __name__ == "__main__":
    main()