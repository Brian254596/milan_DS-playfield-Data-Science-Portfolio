#  Car Price Prediction Using Machine Learning

##  Project Overview

This project builds a machine learning model to predict used car prices based on vehicle specifications such as mileage, horsepower, engine size, age, fuel type, and transmission.

The objective was to identify the most accurate predictive model for estimating car prices and understand the factors that most influence vehicle valuation.

---

##  Business Problem

Used car pricing is influenced by multiple vehicle attributes, making manual valuation inconsistent and subjective.

This project provides a data-driven pricing model that can help:

- Car dealerships estimate resale prices
- Online car marketplaces improve pricing recommendations
- Buyers evaluate fair market value
- Sellers set competitive listing prices

---

## Dataset

The dataset contains **2,000 car records** with the following features:

### Features
- **Brand**
- **Model Year**
- **Engine Size**
- **Mileage**
- **Fuel Type**
- **Transmission**
- **Doors**
- **Owner Count**
- **Horsepower**

### Target Variable
- **Price (USD)**

---

##  Project Workflow

## 1. Business Understanding

Defined the objective of predicting used car prices accurately using machine learning.

---

## 2. Data Understanding

Performed:

- Dataset inspection
- Missing value checks
- Data type validation
- Statistical summary analysis

---

## 3. Exploratory Data Analysis (EDA)

Key analyses performed:

### Distribution Analysis
- Car price distribution

### Correlation Analysis
- Relationship between numerical features and price

### Scatter Plot Analysis
- Mileage vs Price
- Horsepower vs Price
- Engine Size vs Price
- Model Year vs Price

### Key Insights
- Mileage negatively affects price
- Horsepower positively affects price
- Larger engine sizes generally increase price
- Newer cars tend to have higher prices

---

## 4. Feature Engineering

Created additional features to improve predictive performance:

### Car Age
```python
Car_Age = 2026 - Model_Year
```

### Mileage Per Year
```python
Mileage_per_Year = Mileage / (Car_Age + 1)
```

### Power-to-Engine Ratio
```python
Power_to_Engine = Horsepower / Engine_Size
```

### Automatic Transmission Indicator
```python
Is_Automatic
```

---

## 5. Data Preprocessing

Performed:

- Train-test split (80/20)
- One-hot encoding for categorical variables
- Feature scaling handled through pipeline
- Pipeline construction for clean modeling workflow

---

## 6. Modeling

Three models were trained and evaluated:

### 1. Linear Regression
### 2. Random Forest Regressor
### 3. XGBoost Regressor

---

##  Model Performance

| Model | MAE | RMSE | R² Score |
|------|------|------|---------|
| Linear Regression | 1441.45 | 1662.14 | **0.9665** |
| XGBoost | 1750.69 | 2128.56 | 0.9451 |
| Random Forest | 2168.46 | 2681.97 | 0.9128 |

---
## Final Model Selection

## Linear Regression was selected as the final model.

### Why?

Despite being the simplest model, it delivered the best performance:

- **Lowest MAE**
- **Lowest RMSE**
- **Highest R² Score**

### Interpretation

The dataset exhibits a predominantly linear relationship between car features and price.

This demonstrates that:

> Understanding data structure is more important than automatically choosing complex models.

Feature engineering significantly improved linear model performance.

---

##  Key Findings

### Strong Positive Price Drivers
- Horsepower
- Engine Size
- Newer Model Year

### Strong Negative Price Drivers
- Mileage
- Car Age

---

##  Model Deployment

The final model was saved using Joblib:

```python
joblib.dump(model, "best_model.pkl")
```

This allows future predictions without retraining.

---

## 🔧 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib

---

##  Project Structure

```bash
car-price-prediction/
│
├── data/
│   └── car_price_dataset.csv
│
├── notebooks/
│   └── carprice.ipynb
│
├── best_model.pkl
│
└── README.md
```

---

##  Future Improvements

Possible next steps:

- Hyperparameter tuning
- Model deployment with Streamlit
- Real-time price prediction API
- Additional feature enrichment

---

##  Conclusion

This project demonstrates that carefully engineered features combined with proper model evaluation can outperform more complex machine learning algorithms.

Linear Regression proved to be the most effective model for this structured dataset, achieving:

### **R² = 96.65%**

This indicates that the model explains nearly all variability in car prices.

---

##  Author

**Brian Kipchumba**

Applied Statistician | Data Scientist

Focused on applying statistical modeling and machine learning to solve real-world business problems.