import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

file_path = r'C:\Users\Tharun Raman\OneDrive\Documents\GitHub\PRODIGY_ML_01\dataset\train.csv'  # Use raw string for Windows path
df = pd.read_csv(file_path)

features = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
target = df['SalePrice']

if features.isnull().sum().any() or target.isnull().sum() > 0:
    print("Data contains missing values. Please handle them before proceeding.")
else:
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")

    print(f"Intercept: {model.intercept_}")
    print(f"Coefficients: {model.coef_}")

    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs. Predicted Prices")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Line of perfect prediction
    plt.show()
