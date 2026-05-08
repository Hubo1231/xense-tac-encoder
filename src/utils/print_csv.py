"""Pretty-print a CSV file as a terminal table."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


DEFAULT_MAX_ROWS = 50
DEFAULT_MAX_COL_WIDTH = 40
MIN_COL_WIDTH = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read a CSV file and print it as a formatted table.")
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
        "--max-col-width",
        type=int,
        default=DEFAULT_MAX_COL_WIDTH,
        help=f"Maximum width of each column. Default: {DEFAULT_MAX_COL_WIDTH}.",
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
        return [[cell.strip() for cell in row] for row in reader]


def normalize_rows(rows: list[list[str]]) -> list[list[str]]:
    if not rows:
        return []
    width = max(len(row) for row in rows)
    return [row + [""] * (width - len(row)) for row in rows]


def truncate_cell(value: str, width: int) -> str:
    value = value.replace("\n", " ").replace("\r", " ")
    if len(value) <= width:
        return value
    if width <= 3:
        return "." * width
    return value[: width - 3] + "..."


def compute_widths(rows: list[list[str]], *, max_col_width: int, terminal_width: int) -> list[int]:
    if not rows:
        return []

    column_count = len(rows[0])
    widths = [
        min(max_col_width, max(MIN_COL_WIDTH, max(len(row[idx]) for row in rows)))
        for idx in range(column_count)
    ]

    table_overhead = 3 * column_count + 1
    available = max(column_count * MIN_COL_WIDTH, terminal_width - table_overhead)
    total = sum(widths)
    while total > available and any(width > MIN_COL_WIDTH for width in widths):
        widest_idx = max(range(column_count), key=widths.__getitem__)
        widths[widest_idx] -= 1
        total -= 1
    return widths


def border(widths: list[int], left: str = "+", fill: str = "-", joint: str = "+", right: str = "+") -> str:
    return left + joint.join(fill * (width + 2) for width in widths) + right


def format_row(row: list[str], widths: list[int]) -> str:
    cells = [
        f" {truncate_cell(value, width):<{width}} "
        for value, width in zip(row, widths, strict=True)
    ]
    return "|" + "|".join(cells) + "|"


def format_table(rows: list[list[str]], *, no_header: bool, max_rows: int, max_col_width: int) -> str:
    rows = normalize_rows(rows)
    if not rows:
        return "(empty CSV)"

    if no_header:
        header = [f"col_{idx + 1}" for idx in range(len(rows[0]))]
        data_rows = rows
    else:
        header = rows[0]
        data_rows = rows[1:]

    shown_rows = data_rows if max_rows == 0 else data_rows[:max_rows]
    table_rows = [header, *shown_rows]
    terminal_width = shutil.get_terminal_size(fallback=(120, 24)).columns
    widths = compute_widths(table_rows, max_col_width=max(max_col_width, MIN_COL_WIDTH), terminal_width=terminal_width)

    lines = [
        border(widths),
        format_row(header, widths),
        border(widths, fill="="),
    ]
    lines.extend(format_row(row, widths) for row in shown_rows)
    lines.append(border(widths))

    hidden = len(data_rows) - len(shown_rows)
    lines.append(f"rows: {len(data_rows)}" + (f" ({hidden} hidden; use --max-rows 0 to show all)" if hidden > 0 else ""))
    lines.append(f"columns: {len(header)}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.max_rows < 0:
        raise SystemExit("--max-rows must be >= 0")
    if args.max_col_width < 1:
        raise SystemExit("--max-col-width must be >= 1")

    rows = read_csv(args.csv_file, encoding=args.encoding, delimiter=args.delimiter)
    print(
        format_table(
            rows,
            no_header=args.no_header,
            max_rows=args.max_rows,
            max_col_width=args.max_col_width,
        )
    )


if __name__ == "__main__":
    main()
