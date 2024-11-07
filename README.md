# Obesity Classification AI Project

![NumPy](https://img.shields.io/badge/Numpy-v2.1.2-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-v2.2.3-green.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-v3.9.2-orange.svg)
![Seaborn](https://img.shields.io/badge/Seaborn-v0.13.2-yellow.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.5.2-lightblue.svg)
![Joblib](https://img.shields.io/badge/Joblib-v1.4.2-purple.svg)
![Pickle](https://img.shields.io/badge/Pickle-Standard%20Library-lightgrey.svg)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![GitHub issues](https://img.shields.io/github/issues/BytiAX/obesity-classification.svg)](https://github.com/username/repo/issues)

This project aims to classify individuals' obesity levels based on health-related factors, including age, gender, height, weight, and BMI (Body Mass Index). The dataset consists of labeled samples that categorize individuals into different weight classifications.

## Dataset Example

The dataset includes the following columns:

| ID | Age | Gender | Height (cm) | Weight (kg) | BMI  | Label         |
|----|-----|--------|-------------|-------------|------|---------------|
| 1  | 25  | Male   | 175         | 80          | 25.3 | Normal Weight |
| 2  | 30  | Female | 160         | 60          | 22.5 | Normal Weight |

### Column Descriptions
- **ID**: Unique identifier for each individual.
- **Age**: Age of the individual in years.
- **Gender**: Gender of the individual (Male/Female).
- **Height**: Height of the individual in centimeters.
- **Weight**: Weight of the individual in kilograms.
- **BMI**: Body Mass Index, a measure calculated from height and weight.
- **Label**: Class label indicating the weight status (e.g., Normal Weight, Overweight, Obese).

## Project Structure
- **data/**: Folder containing the dataset.
- **notebooks/**: Contains Jupyter Notebook(s) for data analysis, feature engineering, and model training.


## License

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.

**Note:** This project uses the Obesity Classification Dataset from Kaggle. Please ensure compliance with the dataset's license and terms of use: [Kaggle Dataset](https://www.kaggle.com/datasets/sujithmandala/obesity-classification-dataset)
