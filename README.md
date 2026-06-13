# 🏠 House Price Prediction

A Machine Learning project that predicts house prices using Multiple Linear Regression.

## Project Overview

This project uses house-related features such as:

- Square Footage
- Number of Bedrooms
- Number of Bathrooms
- Year Built
- Lot Size
- Garage Size
- Neighborhood Quality

to predict the final house price.

## Models Evaluated

- Multiple Linear Regression ✅ (Selected Final Model)
- Random Forest Regression
- Decision Tree Regression
- Support Vector Regression (SVR)
- Polynomial Regression

## Final Model Performance

| Metric | Value |
|----------|----------|
| R² Score | 0.9984 |
| MAE | ₹ 7,823 |
| RMSE | ₹ 9,882 |
| Accuracy (±10%) | 100% |

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit

## Project Structure

```text
HousePricePrediction/
│
├── app.py
├── house_price_model.pkl
├── house_price_prediction.csv
├── Multiple_linear_regression.ipynb
├── Random_forest_regression.ipynb
├── Decision_tree_regression.ipynb
├── SVR_regression.ipynb
├── Polynomial_linear_regression.ipynb
├── requirements.txt
└── README.md
```

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit app:

```bash
streamlit run app.py
```

## Author

Amit Sharma

B.Tech Student | Machine Learning Enthusiast | Python Developer