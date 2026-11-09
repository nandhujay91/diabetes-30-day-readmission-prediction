import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MAPPING_PATH = PROJECT_ROOT / "data" / "raw" / "IDS_mapping.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "mappings"

MAPPING_COLUMNS = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]


def load_mapping():
    if not MAPPING_PATH.exists():
        raise FileNotFoundError(f"Mapping file not found: {MAPPING_PATH}")

    return pd.read_csv(MAPPING_PATH)


def split_mapping_sections(data):
    sections = {}
    current_section = data.columns[0]

    sections[current_section] = []

    for _, row in data.iterrows():
        category = row.iloc[0]
        description = row.iloc[1]

        if pd.isna(category):
            continue

        category = str(category).strip()

        if category in MAPPING_COLUMNS:
            current_section = category
            sections[current_section] = []
            continue

        if pd.isna(description):
            continue

        sections[current_section].append(
            {
                "id": int(float(category)),
                "description": str(description).strip(),
            }
        )

    return sections


def save_mapping_tables(sections):
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    for name, records in sections.items():
        mapping_table = pd.DataFrame(records)

        output_file = OUTPUT_PATH / f"{name}.csv"

        mapping_table.to_csv(output_file, index=False)

        logger.info(
            "Saved %s with %s categories",
            name,
            len(mapping_table),
        )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    mapping_data = load_mapping()
    sections = split_mapping_sections(mapping_data)

    save_mapping_tables(sections)

    print("\nMAPPING TABLE SUMMARY")

    for name, records in sections.items():
        print(f"{name}: {len(records)} categories")

    print("\nMAPPING TABLES CREATED SUCCESSFULLY")


if __name__ == "__main__":
    main()
