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


## 3. Data Visualization commands in Python

The two major libraries used to create plots are **matplotlib** and **seaborn**. We will learn the prominent plotting functions of both these libraries as applicable to Data Analysis.

Matplotlib:

```python

from matplotlib import pyplot as plt # or
import matplotlib.pyplot as plt
```
Most of the useful plots in this library are in the pyplot subfolder. 

Seaborn:

```python
import seaborn as sns
```
**Matplotlib Functions**
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

## 4. Chi-Square Test for Categorical Variables

The chi-square test is a statistical method used to determine if there is a significant association between two categorical variables. This test is a non-parametric statistical method that evaluates whether the observed frequencies in each category differ significantly from the expected frequencies, assuming no association between the variables.

The test is based on the **chi-square distribution**, which is a family of distributions defined by degrees of freedom (df). These distributions are right-skewed and vary depending on df. A chi-square distribution table lists critical values for given df and significance levels ($\\alpha$), which we use to assess if our computed test statistic is extreme enough to reject the null hypothesis.

**Null Hypothesis and Alternative Hypothesis**

The chi-square test involves formulating two hypotheses:

  * **Null Hypothesis ($H\_{0}$)** - Assumes that there is no association between the categorical variables, implying that any observed differences are due to random chance.
  * **Alternative Hypothesis ($H\_{1}$)** - Assumes that there is a significant association between the variables, indicating that the observed differences are not due to chance alone.

**Formula**

The chi-square statistic is calculated using the formula:
$\\chi^{2}=\\Sigma\\frac{(O\_{i}-E\_{i})^{2}}{E\_{i}}$

where:

  * $O\_{i}$ is the observed frequency for category $i$.
  * $E\_{i}$ is the expected frequency for category $i$, calculated as:
    $E\_{i} = \\frac{(row total \\times column total)}{grand total}$

The sum is taken over all cells in the contingency table.

The calculated chi-square statistic is then compared to a critical value from the chi-square distribution table. The degrees of freedom for the test are calculated as:
$df=(r-1)\\times(c-1)$

where $r$ is the number of rows and $c$ is the number of columns in the table.

**Chi-Square Distribution Table**

A chi-square distribution table provides critical values that vary by degrees of freedom and the significance level ($\\alpha$). These values indicate the threshold **beyond which the test statistic would be considered statistically significant**.

The higher the $\\chi^{2}$ value, the stronger the evidence against $H\_{0}$.

### Python Implementation Example

Below is a Python implementation using `scipy.stats` and `pandas`:

```python
import pandas as pd
from scipy.stats import chi2_contingency

# Create the contingency table
data = [[20, 30],  # Male: [Like, Dislike]
        [25, 25]] # Female: [Like, Dislike]

# Create a DataFrame for clarity
df = pd.DataFrame(data, columns=["Like", "Dislike"], index=["Male", "Female"])

# Perform the Chi-Square Test
chi2, p, dof, expected = chi2_contingency(df)

# Display results
print("Chi-square Statistic:", chi2)
print("Degrees of Freedom:", dof)
print("P-value:", p)
print("Expected Frequencies: \n", expected)
```

**Output:**

```
Chi-square Statistic: 1.008
Degrees of Freedom: 1
P-value: 0.3156
Expected Frequencies: [[22.5 27.5] [22.5 27.5]]
```

**Interpretation:** Since the p-value ($0.3156$) \> 0.05, we fail to reject the null hypothesis, indicating no significant association.

### Applications

1.  **Market Research:** Analyzing the association between customer demographics and product preferences.
2.  **Healthcare:** Studying the relationship between patient characteristics and disease incidence.
3.  **Social Sciences:** Investigating the link between social factors (e.g., education level) and behavioral outcomes (e.g., voting patterns).
4.  **Education:** Examining the connection between teaching methods and student performance.
5.  **Quality Control:** Assessing the association between manufacturing conditions and product defects.

**Practical Example - Strong Association**

Consider a study investigating the relationship between smoking status (smoker, non-smoker) and the incidence of lung disease (disease, no disease). The researcher collects data from 200 individuals and records the following information:

| Category | Disease | No Disease | Total |
|---|---|---|---|
| Smoker | 50 | 30 | 80 |
| Non-Smoker | 20 | 100 | 120 |
| Total | 70 | 130 | 200 |

**Step 1: Calculate Expected Frequencies**
Using the formula for expected frequencies:

$E\_{Smoker, Disease} = \\frac{(80 \\times 70)}{200} = 28$

$E\_{Smoker, No Disease} = \\frac{(80 \\times 130)}{200} = 52$

$E\_{Non-Smoker, Disease} = \\frac{(120 \\times 70)}{200} = 42$

$E\_{Non-Smoker, No Disease} = \\frac{(120 \\times 130)}{200} = 78$

**Step 2: Compute Chi-Square Statistic**
$\\chi^{2} = \\frac{(50-28)^{2}}{28} + \\frac{(30-52)^{2}}{52} + \\frac{(20-42)^{2}}{42} + \\frac{(100-78)^{2}}{78}$

$\\chi^{2} = \\frac{(22)^{2}}{28} + \\frac{(-22)^{2}}{52} + \\frac{(-22)^{2}}{42} + \\frac{(22)^{2}}{78}$

$\\chi^{2} = \\frac{484}{28} + \\frac{484}{52} + \\frac{484}{42} + \\frac{484}{78}$

$\\chi^{2} = 17.29 + 9.31 + 11.52 + 6.21$

$\\chi^{2} = 44.33$

**Step 3: Determine Degrees of Freedom**
$df=(2-1)\\times(2-1)=1$

**Step 4: Interpret the Result**
Using a chi-square distribution table, we compare the calculated chi-square value (44.33) with the critical value at one degree of freedom and a significance level (e.g., 0.05), approximately 3.841. Since $44.33 \> 3.841$, we reject the null hypothesis. This indicates a significant association between smoking status and the incidence of lung disease in this sample.

**Conclusion**

The chi-square test is a powerful tool for analyzing the relationship between categorical variables. By comparing observed and expected frequencies, researchers can determine if there is a statistically significant association, providing valuable insights in various fields of study.
## 5. Correlation in Python
**Key Differences at a Glance**
| Feature | `scipy.stats.pearsonr()` | `pandas.DataFrame.corr()` |
| :--- | :--- | :--- |
| **Primary Use** | Significance test between **two** variables | Generating a correlation matrix for **multiple** variables |
| **Input** | Two 1D arrays/Series | A pandas DataFrame |
| **Output** | Tuple (correlation, p-value) | DataFrame (correlation matrix) |
| **P-value** | **Included** | **Not included** by default |
| **Library** | SciPy | Pandas |


## 6. Model Evaluation - Kernel Density Estimation (KDE) Plots

Kernel Density Estimation (KDE) plots are useful for visualizing data distributions by estimating their probability density function (PDF). In regression analysis, they are particularly effective for comparing actual and predicted values. 

**Why Use KDE Plots?**
1. KDE plots are beneficial for model evaluation for several reasons:
2. They provide a smooth approximation of the data distribution.
3. They are not sensitive to bin sizes, which is a common issue with histograms.
4. They help in effectively comparing true versus predicted distributions.
5. They can highlight deviations between observed and predicted values.

**Implementation in Python**

seaborn.kdeplot() function to compare the distributions of actual and predicted values from a linear regression model.

```python
import numpy as npy
import pandas as pds
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Generating Sample Data
npy.random.seed(42)
x = npy.arange(100)
y = 3 * x + npy.random.normal(0, 10, 100) # Linear relation with noise
data = pds.DataFrame({'X': x, 'Y': y})

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(data[['X']], data['Y'], test_size=0.2, random_state=42)

# Training a Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Plotting KDE for Observed vs. Predicted Values
plt.figure(figsize=(8, 5))
sns.kdeplot(y_test, label='Actual', fill=True, color='blue')
sns.kdeplot(y_pred, label='Predicted', fill=True, color='red')
plt.xlabel('Target Variable')
plt.ylabel('Density')
plt.title('KDE Plot of Actual vs. Predicted Values')
plt.legend()
plt.show()
```
**Interpretation of the KDE Plot**
1. Overlap Between Distributions: A significant overlap between the two curves indicates that the model has captured the general distribution of the actual target values reasonably well.
2. Peak Differences: The actual values (blue curve) may have a higher peak, showing a greater concentration around certain values. 
3. Spread of Distributions: The actual values may have a wider spread, indicating more variation in the real-world data. A narrower spread for predicted values can suggest that the model is underestimating variance, which might be a sign of over-smoothing or bias.
4. Tails of Distributions: If the tails of the predicted values closely follow the actual values, it means the model is not generating extreme outliers. A significant mismatch in the tails could indicate that the model struggles with extreme cases.

## 7. Model Development/Fitting and Model Evaluation

#### 7.1 Linear Regression

| Process | Description | Code Example |
| :--- | :--- | :--- |
| **Create a Linear Regression model object** | | `from sklearn.linear_model import LinearRegression` \
`lr = LinearRegression()` |
| **Train Linear Regression model** | Train the model on decided data, separating Input and Output attributes. When there is a single input attribute, it is simple linear regression. When there are multiple attributes, it is multiple linear regression. | `X = df[['attribute_1', 'attribute_2', ...]] \<br\> Y = df['target_attribute'] \<br\> lr.fit(X,Y)` |
| **Generate output predictions** | Predict the output for a set of Input attribute values. | `Y_hat = lr.predict(X)` |
| **Identify the coefficient and intercept** | Identify the slope coefficient (m) and intercept (c) values of the linear regression model. | `coeff = lr.coef_`\`intercept = lr.intercept_` |
| **Residual Plot** | This function will regress y on x and then draw a scatterplot of the residuals. | `import seaborn as sns`\`sns.residplot(x=df['attribute_1'], y=df ['attribute_2'])` |
| **Distribution Plot** | This function can be used to plot the distribution of data with respect to a given attribute. | `import seaborn as sns`\`sns.distplot(df['attribute_name'], hist=False)` |

#### 7.2 Polynomial Regression

| Process | Description | Code Example |
| :--- | :--- | :--- |
| **For single variable feature creation and model fitting** | Available under the numpy package for single variable feature creation and model fitting[cite: 9]. | `f = np.polyfit(x, y, n)`\<br\`p = np.poly1d(f)`\<br\`Y_hat = p(x)` |
| **Multi-variate Polynomial Regression** | Generate a new feature matrix consisting of all polynomial combinations of the features up to a specified degree[cite: 9]. | `from sklearn.preprocessing import PolynomialFeatures`[cite: 9]\<br\`Z = df[['attribute_1', 'attribute_2',...]]`[cite: 9]\<br\`pr = PolynomialFeatures (degree=n)`[cite: 9]\<br\`Z_pr = pr.fit_transform(Z)` |

#### 7.3 Pipeline

| Process | Description | Code Example |
| :--- | :--- | :--- |
| **Data Pipelines** | Simplify the steps of processing the data by creating a list of tuples with the name of the model/estimator and its corresponding constructor[cite: 9]. | `from sklearn.pipeline import Pipeline` `from sklearn.preprocessing import StandardScaler` `Input=[('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]` `pipe = Pipeline(Input)` `Z = Z.astype(float)`\<br\>`pipe.fit(Z,y)`\<br\>`y_pipe = pipe.predict(Z)` |






#### 7.4 Model Evaluation

| Process | Description | Code Example |
| :--- | :--- | :--- |
| **R² value** | A measure to indicate how close the data is to the fitted regression line[cite: 9]. The value is the percentage of variation of the response variable (y) that is explained by a linear model[cite: 9]. | **For Linear Regression**: \<br\>`X = df[['attribute_1', 'attribute_2', ...]]`\<br\>`Y = df['target_attribute']`\<br\>`lr.fit(X,Y)`\<br\>`R2_score = lr.score(X,Y)` **For Polynomial Regression**: \<br\>`from sklearn.metrics import r2_score`\<br\>`f = np.polyfit(x, y, n)`\<br\>`p = np.poly1d(f)`\<br\>`R2_score = r2_score(y, p(x))` |
| **MSE value** | The Mean Squared Error measures the average of the squares of errors, which is the difference between actual and estimated values[cite: 9]. | `from sklearn.metrics import mean_squared_error` `mse = mean_squared_error(Y, Y_hat)` |