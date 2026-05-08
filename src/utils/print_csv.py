"""Convert a CSV file into LLM-friendly plain text."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_MAX_ROWS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a CSV file and print its content as text suitable for LLM prompts."
    )
    parser.add_argument("csv_file", type=Path, help="Path to the CSV file.")
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="File encoding. Default: utf-8-sig.",
    )
    parser.add_argument(
        "--delimiter",
        default=None,
        help="CSV delimiter. By default it is detected from the file sample.",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Treat the first CSV row as data instead of a header.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help=f"Maximum number of data rows to print. Default: {DEFAULT_MAX_ROWS}. Use 0 for all rows.",
    )
    parser.add_argument(
        "--format",
        choices=("records", "markdown", "jsonl"),
        default="records",
        help="Output format. Default: records.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional title shown at the top of the output.",
    )
    return parser.parse_args()


def detect_dialect(sample: str, delimiter: str | None) -> csv.Dialect:
    if delimiter:
        class ExplicitDialect(csv.excel):
            pass

        ExplicitDialect.delimiter = delimiter
        return ExplicitDialect

    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        return csv.excel


def read_csv(path: Path, *, encoding: str, delimiter: str | None) -> list[list[str]]:
    if not path.is_file():
        raise SystemExit(f"CSV file not found: {path}")

    with path.open("r", encoding=encoding, newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        reader = csv.reader(f, dialect=detect_dialect(sample, delimiter))
        return [[clean_cell(cell) for cell in row] for row in reader]


def clean_cell(value: str) -> str:
    return " ".join(value.replace("\r", "\n").split())


def normalize_rows(rows: list[list[str]]) -> list[list[str]]:
    if not rows:
        return []
    width = max(len(row) for row in rows)
    return [row + [""] * (width - len(row)) for row in rows]


def split_header_and_rows(rows: list[list[str]], *, no_header: bool) -> tuple[list[str], list[list[str]]]:
    rows = normalize_rows(rows)
    if not rows:
        return [], []
    if no_header:
        return [f"column_{idx + 1}" for idx in range(len(rows[0]))], rows
    return rows[0], rows[1:]


def rows_to_dicts(header: list[str], rows: list[list[str]]) -> list[dict[str, str]]:
    seen: dict[str, int] = {}
    keys = []
    for idx, name in enumerate(header):
        key = name or f"column_{idx + 1}"
        if key in seen:
            seen[key] += 1
            key = f"{key}_{seen[key]}"
        else:
            seen[key] = 1
        keys.append(key)
    return [dict(zip(keys, row, strict=True)) for row in rows]


def summarize(path: Path, header: list[str], rows: list[list[str]], shown_count: int, title: str | None) -> list[str]:
    hidden = max(len(rows) - shown_count, 0)
    lines = [
        title or "CSV content",
        f"source: {path}",
        f"columns ({len(header)}): {', '.join(header)}",
        f"total data rows: {len(rows)}",
        f"shown data rows: {shown_count}",
    ]
    if hidden:
        lines.append(f"hidden data rows: {hidden} (use --max-rows 0 to show all)")
    return lines


def format_records(path: Path, header: list[str], rows: list[list[str]], *, max_rows: int, title: str | None) -> str:
    shown_rows = rows if max_rows == 0 else rows[:max_rows]
    records = rows_to_dicts(header, shown_rows)
    lines = summarize(path, header, rows, len(shown_rows), title)
    lines.append("")
    lines.append("Rows:")
    if not records:
        lines.append("(no data rows)")
        return "\n".join(lines)

    for idx, record in enumerate(records, start=1):
        lines.append(f"Row {idx}:")
        for key, value in record.items():
            lines.append(f"- {key}: {value}")
        if idx != len(records):
            lines.append("")
    return "\n".join(lines)


def escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|")


def format_markdown(path: Path, header: list[str], rows: list[list[str]], *, max_rows: int, title: str | None) -> str:
    shown_rows = rows if max_rows == 0 else rows[:max_rows]
    lines = summarize(path, header, rows, len(shown_rows), title)
    lines.append("")
    if not header:
        lines.append("(empty CSV)")
        return "\n".join(lines)

    escaped_header = [escape_markdown_cell(value) for value in header]
    lines.append("| " + " | ".join(escaped_header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in shown_rows:
        lines.append("| " + " | ".join(escape_markdown_cell(value) for value in row) + " |")
    return "\n".join(lines)


def format_jsonl(path: Path, header: list[str], rows: list[list[str]], *, max_rows: int, title: str | None) -> str:
    shown_rows = rows if max_rows == 0 else rows[:max_rows]
    lines = summarize(path, header, rows, len(shown_rows), title)
    lines.append("")
    lines.extend(json.dumps(record, ensure_ascii=False) for record in rows_to_dicts(header, shown_rows))
    if len(lines) == 6:
        lines.append("(no data rows)")
    return "\n".join(lines)


def format_csv_text(
    path: Path,
    rows: list[list[str]],
    *,
    no_header: bool,
    max_rows: int,
    output_format: str,
    title: str | None,
) -> str:
    header, data_rows = split_header_and_rows(rows, no_header=no_header)
    if output_format == "records":
        return format_records(path, header, data_rows, max_rows=max_rows, title=title)
    if output_format == "markdown":
        return format_markdown(path, header, data_rows, max_rows=max_rows, title=title)
    if output_format == "jsonl":
        return format_jsonl(path, header, data_rows, max_rows=max_rows, title=title)
    raise ValueError(f"Unsupported output format: {output_format}")


def main() -> None:
    args = parse_args()
    if args.max_rows < 0:
        raise SystemExit("--max-rows must be >= 0")

    rows = read_csv(args.csv_file, encoding=args.encoding, delimiter=args.delimiter)
    print(
        format_csv_text(
            args.csv_file,
            rows,
            no_header=args.no_header,
            max_rows=args.max_rows,
            output_format=args.format,
            title=args.title,
        )
    )


if __name__ == "__main__":
    main()
