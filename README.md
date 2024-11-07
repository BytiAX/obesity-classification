
# Obesity Classification AI Project

![NumPy](https://img.shields.io/badge/Numpy-v2.1.2-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-v2.2.3-green.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-v3.9.2-orange.svg)
![Seaborn](https://img.shields.io/badge/Seaborn-v0.13.2-yellow.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.5.2-lightblue.svg)
![Joblib](https://img.shields.io/badge/Joblib-v1.4.2-purple.svg)
![Pickle](https://img.shields.io/badge/Pickle-Standard%20Library-lightgrey.svg)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![GitHub issues](https://img.shields.io/github/issues/BytiAX/obesity-classification.svg)](https://github.com/username/repo/issues)

## Introduction

This project classifies individuals' obesity levels based on various health and lifestyle-related attributes, such as age, gender, height, weight, and BMI (Body Mass Index). By utilizing machine learning algorithms, this project aims to predict obesity levels to aid in understanding potential health risks associated with obesity.

## Table of Contents
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Machine Learning Algorithms](#machine-learning-algorithms)
- [Model Evaluation](#model-evaluation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The dataset used in this project comes from the [Obesity Classification Dataset](https://www.kaggle.com/datasets/sujithmandala/obesity-classification-dataset). It includes attributes related to personal characteristics and health status.

### Columns
- **ID**: Unique identifier for each individual
- **Age**: Age in years
- **Gender**: Male or Female
- **Height**: Height in centimeters
- **Weight**: Weight in kilograms
- **BMI**: Body Mass Index
- **Label**: Obesity classification (e.g., Underweight, Normal Weight, Overweight, Obese)

## Data Preprocessing

Data preprocessing steps include:
- **Handling Missing Values**: Ensuring no missing entries in the dataset.
- **Encoding Categorical Variables**: Converting categorical features into numerical formats.
- **Normalizing Numerical Features**: Standardizing numerical values to improve model performance.
- **Splitting the Dataset**: Dividing data into training and testing sets for model validation.

## Exploratory Data Analysis (EDA)

EDA was performed to understand feature distributions and relationships with the target label:
- **Visualizations**: Histograms, box plots, and correlation matrices were used to explore the data.
- **Summary Statistics**: Mean, median, and distribution checks were conducted.

## Machine Learning Algorithms

The following models were evaluated for obesity classification:
- **Linear Support Vector Classifier (SVC)**: Efficient for linearly separable classes, providing quick results for binary or multiclass classification.
- **K-Nearest Neighbors (KNN)**: A simple instance-based classifier, suitable for smaller datasets and capturing local data patterns.
- **Random Forest Classifier**: An ensemble approach that reduces overfitting, effectively handling complex relationships.
- **HistGradientBoosting Classifier**: A sequential boosting model that refines errors, often outperforming simpler classifiers in complex cases.

`Random Forest Classifer` and `HistGradientBoosting Classifier`'s performance was optimized using hyperparameter tuning to achieve the best results for this dataset.

## Model Evaluation

Evaluation metrics include:
- **Accuracy**: The primary metric for overall correctness.
- **Precision, Recall, and F1-Score**: Used to understand model performance on each class.
- **Confusion Matrix**: Provides a detailed view of classification performance across obesity classes.

## Usage

To replicate the analysis and model training:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/otuemre/obesity-classification.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd obesity-classification
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook notebooks/obesity-classification.ipynb
   ```

## Project Structure
- **data/**: Folder containing the dataset.
- **notebooks/**: Contains Jupyter Notebook(s) for data analysis, feature engineering, and model training.
- **images/**: Folder containing visualization graphs generated during data analysis.
- **models/**: Folder where final models are saved as .pkl and .joblib files.
- **README.md**: Project documentation.
- **LICENSE.md**: License information.

## Dependencies

This project relies on the following Python libraries:
- **NumPy**: For numerical operations and array handling.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib** and **Seaborn**: For creating visualizations and plots.
- **Scikit-Learn**: For implementing machine learning algorithms.
- **Joblib**: For saving and loading model files.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.

**Note:** This project uses the Obesity Classification Dataset from Kaggle. Ensure compliance with the dataset's license and terms of use: [Kaggle Dataset](https://www.kaggle.com/datasets/sujithmandala/obesity-classification-dataset).
