import pandas as pd
import numpy as np

def print_dataframe_summary(df, name="DataFrame"):
    """
    Devuelve un sumario completo de un determinado DataFrame.
    """
    
    print("=" * 80)
    print(f"COMPREHENSIVE SUMMARY: {name}".center(80))
    print("=" * 80)
    
    # Información básica
    print("\n" + "─" * 80)
    print("📊 BASIC INFORMATION")
    print("─" * 80)
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Duplicate Rows: {df.duplicated().sum():,} ({df.duplicated().sum()/len(df)*100:.2f}%)")
    
    # Nombres de variables y tipos de datos
    print("\n" + "─" * 80)
    print("📋 COLUMNS AND DATA TYPES")
    print("─" * 80)
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  • {dtype}: {count} column(s)")
    print(f"\nColumn List: {', '.join(df.columns.tolist())}")
    
    # Valores faltantes
    print("\n" + "─" * 80)
    print("❌ MISSING VALUES")
    print("─" * 80)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    }).sort_values('Missing Count', ascending=False)
    
    if missing.sum() == 0:
        print("✓ No missing values found!")
    else:
        print(f"Total Missing Values: {missing.sum():,}")
        print("\nColumns with Missing Values:")
        for col in missing_df[missing_df['Missing Count'] > 0].index:
            count = missing_df.loc[col, 'Missing Count']
            pct = missing_df.loc[col, 'Percentage']
            print(f"  • {col}: {int(count):,} ({pct}%)")
    
    # Primeras filas
    print("\n" + "─" * 80)
    print("🔍 FIRST 5 ROWS")
    print("─" * 80)
    print(df.head().to_string())
    
    # Últimas filas
    print("\n" + "─" * 80)
    print("🔍 LAST 5 ROWS")
    print("─" * 80)
    print(df.tail().to_string())
    
    # Resumen estadístico
    print("\n" + "─" * 80)
    print("📈 STATISTICAL SUMMARY (Numerical Columns)")
    print("─" * 80)
    if len(df.select_dtypes(include=[np.number]).columns) > 0:
        print(df.describe().to_string())
    else:
        print("No numerical columns found.")
    
    # Resumen categórico
    print("\n" + "─" * 80)
    print("🏷️  CATEGORICAL SUMMARY")
    print("─" * 80)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            n_unique = df[col].nunique()
            print(f"\n{col}:")
            print(f"  • Unique Values: {n_unique:,}")
            if n_unique <= 10:
                value_counts = df[col].value_counts()
                print(f"  • Distribution:")
                for val, count in value_counts.items():
                    print(f"    - {val}: {count:,} ({count/len(df)*100:.2f}%)")
            else:
                print(f"  • Top 5 Values:")
                for val, count in df[col].value_counts().head().items():
                    print(f"    - {val}: {count:,} ({count/len(df)*100:.2f}%)")
    else:
        print("No categorical columns found.")
    
    # Información detallada
    print("\n" + "─" * 80)
    print("ℹ️  DETAILED COLUMN INFORMATION")
    print("─" * 80)
    df.info()
    
    print("\n" + "=" * 80)
    print("END OF SUMMARY".center(80))
    print("=" * 80)

