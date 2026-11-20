"""Clean the Dengue monthly dataset and export a model-ready CSV.

Expected input workbook columns (the first worksheet is used):
    Division, Year, Month, Pollution, Density, Infected, Death,
    temperature, Humidity, Pressure, Class

The source workbook uses merged/group-style labels: Division and Year are
written once and left blank on following rows. This script forward-fills
those two columns only, after verifying that a starting value exists.

It does NOT guess missing measurements. Unparseable temperature/pressure
values remain blank in their numeric output columns and are documented in
quality_flag. The original pressure text is preserved in pressure_raw.

Run from the repository root:
    python src/data/clean_data.py

Or specify paths:
    python src/data/clean_data.py --input dataset.xlsx --output dengue_cleaned_dataset.csv

Dependency:
    pip install openpyxl
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from openpyxl import load_workbook


# When this file is saved as src/data/clean_data.py, parents[2] is the
# repository root. The CLI options can override either default path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "dataset.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "dengue_cleaned_dataset.csv"

EXPECTED_SOURCE_COLUMNS = [
    "division",
    "year",
    "month",
    "pollution",
    "density",
    "infected",
    "death",
    "temperature",
    "humidity",
    "pressure",
    "class",
]

OUTPUT_COLUMNS = [
    "year_month",
    "division",
    "year",
    "month",
    "pollution",
    "density",
    "infected",
    "death",
    "temperature_c",
    "humidity",
    "pressure_mbar",
    "class",
    "pressure_raw",
    "quality_flag",
]

TEMPERATURE_PATTERN = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*°\s*C\s*$", re.IGNORECASE)
PRESSURE_PATTERN = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*mbar\s*$", re.IGNORECASE)


def clean_text(value: Any) -> Optional[str]:
    """Trim whitespace (including Excel non-breaking spaces) from text."""
    if value is None:
        return None
    text = str(value).replace("\u00a0", " ").strip()
    return text if text else None


def normalize_header(value: Any) -> str:
    """Normalize a source header so trailing spaces/case do not matter."""
    text = clean_text(value)
    return text.casefold() if text else ""


def parse_year(value: Any, excel_row: int) -> int:
    """Convert an Excel year to an integer without inventing a value."""
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Year is missing and cannot be reconstructed at Excel row {excel_row}.")
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(f"Invalid year {value!r} at Excel row {excel_row}.") from None
    if not numeric.is_integer():
        raise ValueError(f"Year must be a whole number; found {value!r} at Excel row {excel_row}.")
    year = int(numeric)
    if year < 1900 or year > 2100:
        raise ValueError(f"Year {year} is outside the accepted sanity range at Excel row {excel_row}.")
    return year


def parse_month(value: Any, excel_row: int) -> Tuple[str, int]:
    """Return canonical full month name and month number."""
    text = clean_text(value)
    if not text:
        raise ValueError(f"Month is missing at Excel row {excel_row}.")
    try:
        parsed = datetime.strptime(text.title(), "%B")
    except ValueError:
        raise ValueError(f"Unrecognized month {value!r} at Excel row {excel_row}.") from None
    return parsed.strftime("%B"), parsed.month


def parse_integer(value: Any, field_name: str, excel_row: int, flags: List[str]) -> Optional[int]:
    """Parse count columns; leave missing/invalid values blank and flag them."""
    if value is None or clean_text(value) is None:
        flags.append(f"MISSING_{field_name.upper()}")
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        flags.append(f"INVALID_{field_name.upper()}")
        return None
    if not number.is_integer():
        flags.append(f"NON_INTEGER_{field_name.upper()}")
        return None
    result = int(number)
    if result < 0:
        flags.append(f"NEGATIVE_{field_name.upper()}")
    return result


def parse_float(value: Any, field_name: str, flags: List[str]) -> Optional[float]:
    """Parse a numeric measurement without filling missing values."""
    if value is None or clean_text(value) is None:
        flags.append(f"MISSING_{field_name.upper()}")
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        flags.append(f"INVALID_{field_name.upper()}")
        return None


def parse_temperature(value: Any, flags: List[str]) -> Optional[float]:
    """Convert source strings such as '19 °C' or '19 °C' into Celsius."""
    text = clean_text(value)
    if not text:
        flags.append("MISSING_TEMPERATURE")
        return None
    match = TEMPERATURE_PATTERN.fullmatch(text)
    if not match:
        flags.append("REVIEW_TEMPERATURE_FORMAT")
        return None
    return float(match.group(1))


def parse_pressure(value: Any, flags: List[str]) -> Optional[float]:
    """Convert mbar strings to numbers; do not infer unrecognized units."""
    text = clean_text(value)
    if not text:
        flags.append("MISSING_PRESSURE")
        return None
    match = PRESSURE_PATTERN.fullmatch(text)
    if not match:
        flags.append("REVIEW_PRESSURE_UNIT")
        return None
    return float(match.group(1))


def join_flags(flags: Iterable[str]) -> str:
    """Return unique quality flags in their original detection order."""
    return ";".join(dict.fromkeys(flags))


def load_source_rows(input_path: Path) -> Tuple[List[str], List[Tuple[int, Sequence[Any]]]]:
    """Read the first worksheet and retain original Excel row numbers."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input workbook not found: {input_path}\n"
            "Place dataset.xlsx in data/raw/ or pass --input with its location."
        )

    workbook = load_workbook(filename=input_path, read_only=True, data_only=True)
    try:
        worksheet = workbook[workbook.sheetnames[0]]
        iterator = worksheet.iter_rows(values_only=True)
        try:
            raw_headers = next(iterator)
        except StopIteration:
            raise ValueError("The input workbook's first worksheet is empty.") from None

        header_positions: Dict[str, int] = {}
        for index, raw_header in enumerate(raw_headers):
            header = normalize_header(raw_header)
            if header:
                if header in header_positions:
                    raise ValueError(f"Duplicate source column after normalization: {raw_header!r}.")
                header_positions[header] = index

        missing = [name for name in EXPECTED_SOURCE_COLUMNS if name not in header_positions]
        if missing:
            raise ValueError(
                "The workbook is missing required columns: " + ", ".join(missing)
            )

        rows: List[Tuple[int, Sequence[Any]]] = []
        for excel_row, values in enumerate(iterator, start=2):
            # Ignore completely empty worksheet rows, but keep rows with any data.
            if all(value is None or clean_text(value) is None for value in values):
                continue
            rows.append((excel_row, values))

        # Store the source-column index map on the returned header list's
        # companion object would be awkward; return the map as a second item.
        return header_positions, rows
    finally:
        workbook.close()


def clean_dataset(input_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Read, validate, clean and audit the source workbook."""
    column_positions, source_rows = load_source_rows(input_path)

    cleaned_rows: List[Dict[str, Any]] = []
    current_division: Optional[str] = None
    current_year: Optional[int] = None
    seen_keys: Dict[Tuple[str, int, int], Tuple[Any, ...]] = {}
    seen_records = set()
    exact_duplicates_removed = 0
    flagged_rows = 0
    month_coverage: Dict[Tuple[str, int], set] = defaultdict(set)

    def get_value(values: Sequence[Any], column: str) -> Any:
        return values[column_positions[column]]

    for excel_row, values in source_rows:
        # Division and Year are group labels in the supplied workbook. Their
        # blanks are carried forward only after an explicit label was seen.
        raw_division = clean_text(get_value(values, "division"))
        if raw_division:
            current_division = raw_division
        if not current_division:
            raise ValueError(f"Division is blank before the first division label at Excel row {excel_row}.")

        raw_year = get_value(values, "year")
        if raw_year is not None and clean_text(raw_year) is not None:
            current_year = parse_year(raw_year, excel_row)
        if current_year is None:
            raise ValueError(f"Year is blank before the first year label at Excel row {excel_row}.")

        month_name, month_number = parse_month(get_value(values, "month"), excel_row)
        flags: List[str] = []

        pollution = clean_text(get_value(values, "pollution"))
        density = clean_text(get_value(values, "density"))
        class_value = clean_text(get_value(values, "class"))
        if pollution is None:
            flags.append("MISSING_POLLUTION")
        if density is None:
            flags.append("MISSING_DENSITY")
        if class_value is None:
            flags.append("MISSING_CLASS")

        infected = parse_integer(get_value(values, "infected"), "infected", excel_row, flags)
        death = parse_integer(get_value(values, "death"), "death", excel_row, flags)
        temperature_c = parse_temperature(get_value(values, "temperature"), flags)
        humidity = parse_float(get_value(values, "humidity"), "humidity", flags)

        raw_pressure_value = get_value(values, "pressure")
        # Preserve the original pressure cell text verbatim for traceability.
        pressure_raw = None if raw_pressure_value is None else str(raw_pressure_value)
        pressure_mbar = parse_pressure(raw_pressure_value, flags)

        if infected is not None and infected < 0:
            flags.append("NEGATIVE_INFECTED")
        if death is not None and death < 0:
            flags.append("NEGATIVE_DEATH")
        if infected is not None and death is not None and death > infected:
            flags.append("DEATHS_EXCEED_INFECTED")
        if humidity is not None and not 0 <= humidity <= 1:
            flags.append("REVIEW_HUMIDITY_RANGE")

        record: Dict[str, Any] = {
            "year_month": f"{current_year:04d}-{month_number:02d}",
            "division": current_division,
            "year": current_year,
            "month": month_name,
            "pollution": pollution,
            "density": density,
            "infected": infected,
            "death": death,
            "temperature_c": temperature_c,
            "humidity": humidity,
            "pressure_mbar": pressure_mbar,
            "class": class_value,
            "pressure_raw": pressure_raw,
            "quality_flag": join_flags(flags),
        }

        key = (current_division, current_year, month_number)
        record_signature = tuple(record[column] for column in OUTPUT_COLUMNS)

        # Exact duplicate rows can be removed safely. A repeated month with
        # differing values is ambiguous and is stopped for human review.
        if key in seen_keys:
            if seen_keys[key] == record_signature:
                exact_duplicates_removed += 1
                continue
            raise ValueError(
                f"Conflicting duplicate for {current_division}, {current_year}-{month_number:02d} "
                f"at Excel row {excel_row}; review the source rather than choosing a row."
            )

        seen_keys[key] = record_signature
        if record_signature in seen_records:
            # A fully repeated row should already have the same monthly key;
            # this branch catches an identical record with an inconsistent key.
            flags.append("REPEATED_RECORD_DIFFERENT_DATE")
            record["quality_flag"] = join_flags(flags)
        seen_records.add(record_signature)

        if record["quality_flag"]:
            flagged_rows += 1
        cleaned_rows.append(record)
        month_coverage[(current_division, current_year)].add(month_number)

    # Preserve the source workbook order (division/year/month) for traceability.

    incomplete_periods = []
    for (division, year), observed_months in sorted(month_coverage.items()):
        missing_months = sorted(set(range(1, 13)) - observed_months)
        if missing_months:
            incomplete_periods.append({
                "division": division,
                "year": year,
                "missing_month_numbers": missing_months,
            })

    audit = {
        "source_rows": len(source_rows),
        "cleaned_rows": len(cleaned_rows),
        "exact_duplicates_removed": exact_duplicates_removed,
        "flagged_rows": flagged_rows,
        "divisions": sorted({row["division"] for row in cleaned_rows}),
        "years": sorted({row["year"] for row in cleaned_rows}),
        "incomplete_division_year_periods": incomplete_periods,
    }
    return cleaned_rows, audit


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Write the cleaned records using the agreed output column order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean the dengue Excel dataset and export a processed CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input .xlsx workbook (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output .csv path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    try:
        cleaned_rows, audit = clean_dataset(input_path)
        write_csv(cleaned_rows, output_path)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("Dengue dataset cleaning completed.")
    print(f"Input workbook:             {input_path}")
    print(f"Output CSV:                 {output_path}")
    print(f"Source data rows:           {audit['source_rows']}")
    print(f"Cleaned data rows:          {audit['cleaned_rows']}")
    print(f"Exact duplicate rows removed: {audit['exact_duplicates_removed']}")
    print(f"Rows with quality flags:    {audit['flagged_rows']}")
    print(f"Divisions:                  {', '.join(audit['divisions'])}")
    if audit["years"]:
        print(f"Year range:                 {audit['years'][0]}–{audit['years'][-1]}")
    if audit["incomplete_division_year_periods"]:
        print("WARNING: Some division/year groups do not contain all 12 months:")
        for item in audit["incomplete_division_year_periods"]:
            print(f"  - {item['division']} {item['year']}: missing {item['missing_month_numbers']}")
    else:
        print("Monthly coverage:           all observed division/year groups have 12 months")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
