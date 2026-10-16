
import pandas as pd


def print_dataframe_summary(
    df: pd.DataFrame,
    name: str="DataFrame"
) -> None:
    """Prints a complete summary of a DataFrame

    Args:
        df (pandas.DataFrame): base DataFrame
        name (str, optional): name of the DataFrame. Defaults to "DataFrame".

    Returns:
        None
    """

    width = 80
    rows, cols = df.shape

    def section(title):
        print(f"\n{'─' * width}")
        print(title)
        print("─" * width)

    def percentage(count, total):
        return count / total * 100 if total else 0.0

    def print_frequency(series, total):
        for value, count in series.items():
            pct = percentage(count, total)
            print(f"    - {value}: {count:,} ({pct:.2f}%)")

    print("=" * width)
    print(f"SUMMARY: {name}".center(width))
    print("=" * width)

    section("📊 BASIC INFORMATION")
    print(f"Shape: {rows:,} rows × {cols} columns")
    memory = df.memory_usage(deep=True).sum()
    memory_mb = memory / (1024 ** 2)
    print(f"Memory Usage: {memory_mb:.2f} MB")

    duplicates = int(df.duplicated().sum())
    dup_pct = percentage(duplicates, rows)
    print(f"Duplicate Rows: {duplicates:,} ({dup_pct:.2f}%)")

    section("📋 COLUMNS AND DATA TYPES")
    dtype_counts = df.dtypes.value_counts()

    for dtype, count in dtype_counts.items():
        print(f"  • {dtype}: {count} column(s)")

    print(f"\nColumn List: {', '.join(map(str, df.columns))}")

    section("❌ MISSING VALUES")
    missing = df.isna().sum()
    total_missing = int(missing.sum())
    missing_pct = missing.div(rows).mul(100).round(2)

    if total_missing == 0:
        print("✓ No missing values found!")
    else:
        print(f"Total Missing Values: {total_missing:,}")
        print("\nColumns with Missing Values:")

        missing_cols = missing[missing.gt(0)].index

        for col in missing_cols:
            count = int(missing[col])
            pct = missing_pct[col]
            print(f"  • {col}: {count:,} ({pct:.2f}%)")

    section("🔍 FIRST 5 ROWS")
    print(df.head().to_string())
    section("🔍 LAST 5 ROWS")
    print(df.tail().to_string())

    section("📈 STATISTICAL SUMMARY")
    numeric = df.select_dtypes(include="number")

    if not numeric.empty and rows:
        print(numeric.describe().to_string())
    else:
        print("No numerical data available.")

    # Resumen categórico
    section("🏷️ CATEGORICAL SUMMARY")

    categorical = df.select_dtypes(
        include=["object", "category", "string", "bool"]
    )

    if categorical.shape[1] == 0:
        print("No categorical columns found.")

    for col in categorical.columns:
        series = df[col]
        unique = series.nunique()
        print(f"\n{col}:")
        print(f"  • Unique Values: {unique:,}")

        counts = series.value_counts(dropna=False)

        if unique <= 10:
            print("  • Distribution:")
            print_frequency(counts, rows)
        else:
            print("  • Top 5 Values:")
            print_frequency(counts.head(5), rows)

    section("ℹ️ DETAILED COLUMN INFORMATION")
    df.info()

    print("\n" + "=" * width)
    print("END OF SUMMARY".center(width))
    print("=" * width)
