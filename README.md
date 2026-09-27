# -Integrative-Capstone-Project-and-Evaluation
An end-to-end house price prediction project using Python and machine learning. The project covers data cleaning, preprocessing, exploratory data analysis, correlation and outlier analysis, feature analysis, Linear Regression and Random Forest modeling, model evaluation, feature importance, prediction error analysis, and insights.


# 🏠 House Price Prediction Using Machine Learning

## Project Overview

This project is an end-to-end **House Price Prediction** machine learning project developed using Python as part of an Integrative Capstone Project for a Data Science with Python internship.

The objective is to analyze housing data, identify important factors associated with house prices, build regression models, and evaluate their predictive performance.

The project follows a complete data science workflow, including data acquisition, data preprocessing, exploratory data analysis, feature analysis, model development, evaluation, and interpretation.

---

##  Objectives

The main objectives of this project are:

* Load and inspect a publicly available housing dataset.
* Perform data cleaning and preprocessing.
* Handle categorical variables using encoding techniques.
* Perform exploratory data analysis.
* Analyze relationships between house prices and property features.
* Identify potential outliers.
* Build machine learning regression models.
* Compare model performance using appropriate evaluation metrics.
* Identify important features influencing predictions.
* Analyze prediction errors.
* Provide insights, recommendations, limitations, and future scope.

---

##  Dataset

The dataset used in this project is a **House Price dataset obtained from Kaggle**.

The dataset is provided under the **CC0: Public Domain** license.

### Main Features

| Feature            | Description                    |
| ------------------ | ------------------------------ |
| `price`            | House price / target variable  |
| `area`             | Area of the property           |
| `bedrooms`         | Number of bedrooms             |
| `bathrooms`        | Number of bathrooms            |
| `stories`          | Number of stories              |
| `mainroad`         | Main road access               |
| `guestroom`        | Availability of guest room     |
| `basement`         | Availability of basement       |
| `hotwaterheating`  | Hot water heating availability |
| `airconditioning`  | Air conditioning availability  |
| `parking`          | Number of parking spaces       |
| `prefarea`         | Preferred-area status          |
| `furnishingstatus` | Furnishing condition           |

---

##  Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Jupyter Notebook / Google Colab**

---

##  Project Workflow

The project follows the following pipeline:

```text
Data Acquisition
       ↓
Data Inspection
       ↓
Data Cleaning
       ↓
Categorical Encoding
       ↓
Exploratory Data Analysis
       ↓
Correlation Analysis
       ↓
Outlier Analysis
       ↓
Train-Test Split
       ↓
Model Development
       ↓
Model Evaluation
       ↓
Feature Importance
       ↓
Prediction Error Analysis
       ↓
Insights & Recommendations
```

---

##  Data Preprocessing

The dataset was initially inspected using:

```python
df.head()
df.info()
df.describe()
```

Missing values and duplicate records were checked.

Binary categorical variables such as `yes` and `no` were converted into numerical values:

```python
df[col] = df[col].map({'yes': 1, 'no': 0})
```

The `furnishingstatus` variable was converted using one-hot encoding:

```python
df = pd.get_dummies(
    df,
    columns=['furnishingstatus'],
    drop_first=True
)
```

---

##  Exploratory Data Analysis

Several visualizations were created to understand the dataset:

* House price distribution
* Numerical feature distributions
* Correlation heatmap
* Price and area boxplots
* Actual vs predicted prices
* Prediction error distribution
* Model performance comparison
* Random Forest feature importance

### Correlation Analysis

The strongest positive correlations with house price included:

| Feature          | Correlation |
| ---------------- | ----------: |
| Area             |       0.536 |
| Bathrooms        |       0.518 |
| Air Conditioning |       0.453 |
| Stories          |       0.421 |
| Parking          |       0.384 |
| Bedrooms         |       0.366 |

The results suggest that property area and number of bathrooms have relatively strong relationships with house price in this dataset.

---

##  Outlier Analysis

The IQR method was used to identify potential outliers.

The analysis identified:

* **15 potential outliers in price**
* **12 potential outliers in area**

These observations were retained because unusually expensive or large houses may represent legitimate properties rather than incorrect data.

---

##  Machine Learning Models

Two supervised regression models were developed.

### 1. Linear Regression

Linear Regression was used as a baseline model because it provides a simple and interpretable approach to predicting continuous values.

### 2. Random Forest Regression

Random Forest Regression was implemented using:

```python
RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
```

Random Forest was selected because it can capture nonlinear relationships between features and the target variable.

---

##  Model Evaluation

The models were evaluated using:

* **MAE — Mean Absolute Error**
* **RMSE — Root Mean Squared Error**
* **R² Score**

### Results

| Model             |       MAE |      RMSE | R² Score |
| ----------------- | --------: | --------: | -------: |
| Linear Regression |   970,043 | 1,324,507 |    0.653 |
| Random Forest     | 1,022,560 | 1,401,497 |    0.611 |

For the selected train-test split, Linear Regression produced lower MAE and RMSE and a higher R² score than the configured Random Forest model.

The Linear Regression model achieved an R² score of approximately **0.653**, indicating that it explained about 65.3% of the variation in house prices in the test dataset.

---

##  Feature Importance

Random Forest feature importance showed that the most influential features were:

 FeatureRandom Forest feature importance showed that the most influential features were:

Feature	Importance
Area	0.468
Bathrooms	0.152
Air Conditioning	0.063
Parking	0.058
Stories	0.057

The area feature had the highest feature importance in the Random Forest model.

# Prediction Error Analysis

Prediction errors were calculated by subtracting predicted prices from actual prices.

For the 109 test observations:

Mean error: approximately 146,055
Median error: approximately -81,899
Minimum error: approximately -2,603,188
Maximum error: approximately 5,331,724
Standard deviation: approximately 1,322,510

The error analysis shows that although the model performs reasonably well overall, some individual properties have relatively large prediction errors.

## Key Insights
Area showed the strongest correlation with house price among the analyzed variables.
Bathrooms also showed a relatively strong positive relationship with price.
Features such as air conditioning, parking, stories, and preferred-area status contributed to the prediction process.
House prices contain substantial variability that cannot be fully explained by the available features.
A more complex model does not necessarily provide better results; in this experiment, Linear Regression performed better on the selected test split.
## Limitations
The dataset contains a limited number of property characteristics.
Location-specific information is limited.
Only two machine learning algorithms were evaluated.
Hyperparameter tuning was not extensively performed.
A single train-test split was used instead of cross-validation.
External economic and housing-market factors were not included.
Model predictions should not be considered guaranteed real-world property valuations.
## Future Scope

Future improvements could include:

Applying cross-validation.
Performing hyperparameter optimization.
Testing additional regression algorithms.
Exploring Gradient Boosting and XGBoost.
Performing additional feature engineering.
Adding geographical and neighborhood information.
Using larger and more recent housing datasets.
Developing an interactive web application for house price prediction.

## Conclusion

This project demonstrates a complete data science and machine learning workflow for house price prediction using Python.

The project successfully covers data acquisition, preprocessing, exploratory analysis, feature analysis, model development, evaluation, and interpretation.

Two regression models were implemented and compared. Based on the selected test split, Linear Regression achieved an R² score of approximately 0.653, while Random Forest achieved an R² score of approximately 0.611.

The project also demonstrates the importance of analyzing model errors and understanding feature contributions rather than relying solely on a single performance metric.

Overall, the project provides practical experience in applying Python and machine learning techniques to a real-world regression problem.
