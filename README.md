# Data Visualization using Python

## 1. Data Import & Output Summary

**File I/O Operations**
| Format  | Read Method               | Write Method              | Key Parameters                     |
|---------|---------------------------|---------------------------|------------------------------------|
| CSV     | `pd.read_csv()`           | `df.to_csv()`             | `sep`, `index`, `encoding`         |
| Excel   | `pd.read_excel()`         | `df.to_excel()`           | `sheet_name`, `index`              |
| JSON    | `pd.read_json()`          | `df.to_json()`            | `orient`, `lines`                  |
| Parquet | `pd.read_parquet()`       | `df.to_parquet()`         | `engine` (`pyarrow`/`fastparquet`) |
| SQL     | `pd.read_sql()`           | `df.to_sql()`             | `con` (DB connection), `if_exists` |

**Example Usage**
```python
# Reading Data
# CSV
df = pd.read_csv('data.csv', sep=',', encoding='utf-8')

# Excel
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# JSON
df = pd.read_json('data.json', orient='records')

# Writing Data
# CSV (no index)
df.to_csv('output.csv', index=False)

# Excel (multiple sheets)
with pd.ExcelWriter('output.xlsx') as writer:
    df.to_excel(writer, sheet_name='Sheet1')

# Parquet (compressed)
df.to_parquet('output.parquet', engine='pyarrow', compression='gzip')
```

## 2. Data Pre-processing/Wrangling

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

**Usage Examples**

```python
# Basic Missing Value Count
import pandas as pd

df = pd.DataFrame({'A': [1, None, 3], 'B': ['x', None, 'z']})
print(df.isna().sum())
```


### Data Wrangling

**Key Parameters & Methods**

| Method/Parameter       | Description                                  | Example                          |
|------------------------|----------------------------------------------|----------------------------------|
| Replace missing data with frequency        |  | `MostFrequentEntry = df['attribute_name'].value_counts().idxmax()\ df['attribute_name'].replace(np.nan,MostFrequentEntry,inplace=True)`        |
| Binning      | 	Create bins of data for better analysis and visualization.         | `bins = np.linspace(min(df['attribute_name']), max(df['attribute_name']),n) \ GroupNames = ['Group1','Group2','Group3,...] \ df['binned_attribute_name'] = pd.cut(df['attribute_name'], bins, labels=GroupNames, include_lowest=True)`           |
| Change column name    | | `df.rename(columns={'old_name':\'new_name'}, inplace=True)`          |
| Indicator Variables          | Create indicator variables for categorical data. | `dummy_variable = pd.get_dummies(df['attribute_name']) \ df = pd.concat([df, dummy_variable],axis = 1)` |
