# Data Visualization using Python

## Data Importing
## 2. Data Import & Output Summary

**File I/O Operations**
| Format  | Read Method               | Write Method              | Key Parameters                     |
|---------|---------------------------|---------------------------|------------------------------------|
| CSV     | `pd.read_csv()`           | `df.to_csv()`             | `sep`, `index`, `encoding`         |
| Excel   | `pd.read_excel()`         | `df.to_excel()`           | `sheet_name`, `index`              |
| JSON    | `pd.read_json()`          | `df.to_json()`            | `orient`, `lines`                  |
| Parquet | `pd.read_parquet()`       | `df.to_parquet()`         | `engine` (`pyarrow`/`fastparquet`) |
| SQL     | `pd.read_sql()`           | `df.to_sql()`             | `con` (DB connection), `if_exists` |

**Example Usage**
Reading Data
```python
# CSV
df = pd.read_csv('data.csv', sep=',', encoding='utf-8')

# Excel
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# JSON
df = pd.read_json('data.json', orient='records')

# CSV (no index)
df.to_csv('output.csv', index=False)

# Excel (multiple sheets)
with pd.ExcelWriter('output.xlsx') as writer:
    df.to_excel(writer, sheet_name='Sheet1')

# Parquet (compressed)
df.to_parquet('output.parquet', engine='pyarrow', compression='gzip')
```

## Data Pre-processing/Wrangling

### Pandas Missing Data Handling Guide

This guide covers essential methods and parameters for handling missing data in pandas DataFrames, focusing on column-wise operations.

**Key Parameters & Methods**

| Method/Parameter       | Description                                  | Example                          |
|------------------------|----------------------------------------------|----------------------------------|
| `.isna().sum()`        | Count missing values per column.             | `df['col'].isna().sum()`        |
| `.fillna(value)`       | Fill missing values with a constant.         | `df['col'].fillna(0)`           |
| `.mean()/.median()`    | Fill with mean/median (numeric columns).     | `df.fillna(df.mean())`          |
| `.mode()[0]`           | Fill with most frequent value (categorical). | `df['col'].fillna(df['col'].mode()[0])` |
| `.dropna()`            | Drop rows/columns with missing values.       | `df.dropna(axis=1, thresh=0.5*len(df))` |

## Usage Examples

### 1. Basic Missing Value Count
```python
import pandas as pd

df = pd.DataFrame({'A': [1, None, 3], 'B': ['x', None, 'z']})
print(df.isna().sum())
```