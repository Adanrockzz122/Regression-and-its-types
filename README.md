# Regression and Its Types: California Housing Prices Analysis

This repository contains a Jupyter Notebook (`regression-and-its-types.ipynb`) that demonstrates various regression techniques for predicting median house values using the California Housing Prices dataset. The notebook provides a comprehensive exploration of regression models, including Linear Regression, Polynomial Regression, Ridge Regression, Lasso Regression, Decision Tree Regression, and Random Forest Regression, along with evaluation metrics and visualizations.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Notebook Structure](#notebook-structure)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The notebook implements and compares multiple regression algorithms to predict the median house value (`median_house_value`) based on various features from the California Housing Prices dataset. It includes data preprocessing, model training, evaluation, and visualization of results. The primary goal is to illustrate the application of different regression techniques and their performance on a real-world dataset.

Key objectives:
- Understand and implement various regression models.
- Preprocess the dataset, including handling missing values and categorical variables.
- Evaluate model performance using Mean Squared Error (MSE) and R² Score.
- Visualize actual vs. predicted values for each model.

## Dataset
The notebook uses the **California Housing Prices** dataset, which is publicly available and commonly used for regression tasks. The dataset contains information about housing in California, with the following features:
- **longitude**: Longitude of the house location.
- **latitude**: Latitude of the house location.
- **housing_median_age**: Median age of houses in the area.
- **total_rooms**: Total number of rooms in the area.
- **total_bedrooms**: Total number of bedrooms in the area.
- **population**: Population in the area.
- **households**: Number of households in the area.
- **median_income**: Median income of residents in the area.
- **ocean_proximity**: Categorical variable indicating proximity to the ocean (e.g., NEAR BAY, INLAND).
- **median_house_value**: Target variable (continuous) representing the median house value in USD.

The dataset is loaded from a CSV file (`housing.csv`) available in the Kaggle environment or included in the repository as a data source.

### Data Preprocessing
- **Missing Values**: Rows with missing values are dropped using `df.dropna()`.
- **Categorical Encoding**: The `ocean_proximity` column is one-hot encoded using `pd.get_dummies()`.
- **Feature Scaling**: Features are standardized using `StandardScaler` to ensure consistent scales across variables.
- **Train-Test Split**: The dataset is split into 80% training and 20% testing sets with a fixed random seed (`random_state=42`) for reproducibility.

## Dependencies
To run the notebook, ensure you have the following Python libraries installed:
- `pandas` (for data manipulation)
- `numpy` (for numerical operations)
- `matplotlib` (for plotting)
- `seaborn` (for enhanced visualizations)
- `scikit-learn` (for machine learning models and metrics)

You can install these dependencies using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
