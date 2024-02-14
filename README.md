# Air Quality Index Prediction Project

## Introduction:
This project is dedicated to forecasting the Air Quality Index (AQI) through advanced machine learning techniques. Leveraging the City Day dataset, we explore a range of air quality-related features to enhance predictions and contribute to effective air quality management.
Project URL:  [![Streamlit](https://img.shields.io/badge/Streamlit-%230077B5.svg?logo=streamlit&logoColor=white)](https://aqi-predict-app.onrender.com/)

```python
# AQI Features: 
# PM2.5: Particulate Matter 2.5 micrometers or smaller
# PM10: Particulate Matter 10 micrometers or smaller
# NO: Nitric Oxide
# NO2: Nitrogen Dioxide
# NOx: Nitrogen Oxides (NO + NO2)
# NH3: Ammonia
# CO: Carbon Monoxide
# SO2: Sulfur Dioxide
# O3: Ozone
# Benzene: Benzene, a volatile organic compound (VOC)
# Toluene: Toluene, a volatile organic compound (VOC)
```

## Project Flow:

### Data Loading:
1. **Load the dataset:**
   ```python
   import pandas as pd
   df = pd.read_csv("city_day.csv")
   ```
   Utilize Pandas to load the dataset from the provided CSV file.

2. **Explore data structure:**
   ```python
   df.head()
   ```
   Display the top 5 rows to understand the dataset's structure.

### Exploratory Data Analysis: 
1. **Air Quality Index Category Distribution:**
![image](https://github.com/Surendraprajapat18/Air-Quality-Index-Prediction-/assets/97840357/63ad1e9f-0d43-4791-bbd5-b2c5dba45d23)

2. **Feature Correlation:**
   ![image](https://github.com/Surendraprajapat18/Air-Quality-Index-Prediction-/assets/97840357/f6ea7524-aa6e-4f9f-bc40-af8a00855846)


### Data Preprocessing:
1. **Drop columns with missing values:**
   ```python
   df = df.dropna(axis=1, thresh=len(df)*0.5)
   ```
   Eliminate columns with over 50% missing values and those deemed unnecessary.

2. **Handle null values:**
   ```python
   df = df.fillna(df.mean())
   ```
   Replace null values with the mean to maintain data integrity.

3. **Outlier detection and handling:**
   ```python
   Q1 = df.quantile(0.25)
   Q3 = df.quantile(0.75)
   IQR = Q3 - Q1
   df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

   ```
   ![image](https://github.com/Surendraprajapat18/Air-Quality-Index-Prediction-/assets/97840357/c2d9442b-90fd-4d18-8d22-19904e4a25c6)
   Use the Interquartile Range (IQR) method to effectively identify and address outliers.

### Feature Engineering:
1. **Outlier analysis:**
   ```python
   for column in df.columns:
       IQR = df[column].quantile(0.75) - df[column].quantile(0.25)
       lower = df[column].quantile(0.25) - (IQR * 1.5)
       upper = df[column].quantile(0.75) + (IQR * 1.5)
       outliers_percentage = (len(df[(df[column] < lower) | (df[column] > upper)]) / len(df)) * 100
       print(f'The percentage of outliers in {column}: {outliers_percentage}%')
   ```
   Examine the percentage of outliers in each feature to comprehend data distribution.

2. **Compute IQR and handle outliers:**
   ```python
   Q1 = df.quantile(0.25)
   Q3 = df.quantile(0.75)
   IQR = Q3 - Q1
   df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
   ```
   Compute the Interquartile Range (IQR) to identify and handle outliers effectively.

3. **Multy model Evaluation:**

   The performance metrics for different models are as follows:
     ```python
   models = {
    "LR": LinearRegression(),
    "GrandBoostReg": GradientBoostingRegressor(),
    "KNR": KNeighborsRegressor(),
    "rfr": RandomForestRegressor()
    }
   ```

   **Model Evaluation Results:**

    R2 Score:
      ```python
    {'LR': 0.8356026560032711,
   'GrandBoostReg': 0.8674279441387858,
   'KNR': 0.8588734538154051,
   'rfr': 0.8747495088245665}
    ```

**Note: The provided code snippets are illustrative and should be adapted based on the dataset's structure and requirements and this is Capstone Project.**
