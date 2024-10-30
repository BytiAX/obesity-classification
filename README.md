# Obesity Classification AI Project

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
