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

### 2.1 Pandas Missing Data Handling Guide

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


### 2.2 Data Wrangling

**Key Parameters & Methods**

| Method/Parameter       | Description                                  | Example                          |
|------------------------|----------------------------------------------|----------------------------------|
| Replace missing data with frequency        |  | `MostFrequentEntry = df['attribute_name'].value_counts().idxmax()\ df['attribute_name'].replace(np.nan,MostFrequentEntry,inplace=True)`        |
| Binning      | 	Create bins of data for better analysis and visualization.         | `bins = np.linspace(min(df['attribute_name']), max(df['attribute_name']),n) \ GroupNames = ['Group1','Group2','Group3,...] \ df['binned_attribute_name'] = pd.cut(df['attribute_name'], bins, labels=GroupNames, include_lowest=True)`           |
| Change column name    | | `df.rename(columns={'old_name':\'new_name'}, inplace=True)`          |
| Indicator Variables          | Create indicator variables for categorical data. | `dummy_variable = pd.get_dummies(df['attribute_name']) \ df = pd.concat([df, dummy_variable],axis = 1)` |


### 2.3 Data Visualization commands in Python

The two major libraries used to create plots are **matplotlib** and **seaborn**. We will learn the prominent plotting functions of both these libraries as applicable to Data Analysis.

**Matplotlib**:

```python

from matplotlib import pyplot as plt # or
import matplotlib.pyplot as plt
```
Most of the useful plots in this library are in the pyplot subfolder. 

Seaborn:

```python
import seaborn as sns
```
Matplotlib Functions
1. Standard Line Plot
2. Scatter Plot
3. Histogram
4. Bar Plot
5. Pseudo Color Plot
A pseudocolor plot displays matrix data as an array of colored cells (faces). It is created on an x-y plane defined by a grid of x and y coordinates. A matrix C specifies the colors at the vertices.
```python
plt.plot(x, y)
plt.scatter(x, y)
plt.hist(x, bins)
# You can use the edgecolor argument for better clarity.
plt.bar(x, height)
plt.pcolor(C)
```

**Seaborn Functions**
1. Regression Plot
2. Box and Whisker Plot
3. Residual Plot
   Residuals are the differences between the observed values of the dependent variable and the predicted values from the regression model. They measure how much a regression line misses a data point vertically, indicating the accuracy of the predictions.
4. KDE Plot
   A Kernel Density Estimate (KDE) plot creates a probability distribution curve for a single vector of information based on the likelihood of a specific value's occurrence.
5. Distribution Plot
   This plot combines the histogram and KDE plots. You can choose to display the histogram along with the distribution plot or not. 

```python
sns.regplot(x='header_1', y='header_2', data=df)
sns.residplot(data=df, x='header_1', y='header_2')
sns.residplot(x=df['header_1'], y=df['header_2'])
sns.kdeplot(X)
sns.distplot(X, hist=False)
```