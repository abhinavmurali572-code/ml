# ML Assignment 2: Regression - California Housing Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============== 1. LOADING AND PREPROCESSING (2 marks) ==============

# Load California Housing dataset
california = fetch_california_housing()
print(california.DESCR)  # Dataset description

# Convert to DataFrame
df = pd.DataFrame(california.data, columns=california.feature_names)
df['MedHouseVal'] = california.target  # Target = Median House Value
print(f"Shape: {df.shape}")
print(df.head())

print("\n--- Info ---")
print(df.info())
print(df.describe())

# Handle missing values (check - this dataset has no nulls, but we implement handling)
print("\nMissing values:\n", df.isnull().sum())
# If any missing: df.fillna(df.median(), inplace=True)
# Or: from sklearn.impute import SimpleImputer

# Feature Scaling - Standardization (Necessary because features have different scales)
# Eg: MedInc ~0-15, HouseAge ~1-52, AveRooms ~0-20 etc.
# Scaling is critical for SVR and improves Gradient Boosting convergence

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Train-test split BEFORE scaling to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled back to DF for easy viewing
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

print("\n--- After StandardScaler ---")
print(X_train_scaled_df.describe())

# Preprocessing Justification:
# 1. Standardization: Features like Latitude/Longitude and MedInc have different ranges.
#    SVR calculates distances, so without scaling it will be biased.
# 2. No missing value handling needed as dataset is clean, but we verified it.
# 3. Train-test split with random_state=42 ensures reproducibility.

# ============== 2. REGRESSION ALGORITHM IMPLEMENTATION (5 marks) ==============

# Dictionary to store results
results = {}

# 1. Linear Regression
# How it works: Assumes linear relationship y = wX + b, minimizes squared error using OLS.
# Suitability: Good baseline, interpretable, works well if features are linearly correlated with price.
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
results['Linear Regression'] = y_pred_lr

# 2. Decision Tree Regressor
# How it works: Splits data recursively based on feature thresholds to minimize MSE in leaf nodes.
# Suitability: Captures non-linear relationships, handles feature interactions, no need for feature scaling but we already did.
dt = DecisionTreeRegressor(random_state=42, max_depth=10)
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)
results['Decision Tree'] = y_pred_dt

# 3. Random Forest Regressor
# How it works: Ensemble of Decision Trees (Bagging). Averages predictions of many trees trained on random subsets.
# Suitability: Reduces overfitting of single tree, very robust for housing data with non-linear interactions.
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
results['Random Forest'] = y_pred_rf

# 4. Gradient Boosting Regressor
# How it works: Boosting technique - builds trees sequentially, each new tree corrects errors of previous trees.
# Suitability: Often best for tabular regression, gives high accuracy for housing price prediction.
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train_scaled, y_train)
y_pred_gb = gb.predict(X_test_scaled)
results['Gradient Boosting'] = y_pred_gb

# 5. Support Vector Regressor (SVR)
# How it works: Finds hyperplane that fits data within epsilon margin, uses kernel trick for non-linearity.
# Suitability: Effective for high-dimensional data, but sensitive to scaling (hence standardization is mandatory).
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr.fit(X_train_scaled, y_train)
y_pred_svr = svr.predict(X_test_scaled)
results['SVR'] = y_pred_svr

# ============== 3. MODEL EVALUATION AND COMPARISON (2 marks) ==============

def evaluate_model(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name} -> MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return [mse, mae, r2]

print("\n--- MODEL EVALUATION ---")
metrics_df = pd.DataFrame(columns=['Model', 'MSE', 'MAE', 'R2'])

for model_name, y_pred in results.items():
    mse, mae, r2 = evaluate_model(y_test, y_pred, model_name)
    metrics_df.loc[len(metrics_df)] = [model_name, mse, mae, r2]

print("\n--- Comparison Table ---")
print(metrics_df.sort_values(by='R2', ascending=False))

# Visualization
plt.figure(figsize=(10,5))
sns.barplot(data=metrics_df, x='Model', y='R2')
plt.title('R2 Score Comparison of Regression Models')
plt.xticks(rotation=15)
plt.show()

plt.figure(figsize=(10,5))
sns.barplot(data=metrics_df, x='Model', y='MSE')
plt.title('MSE Comparison (Lower is Better)')
plt.xticks(rotation=15)
plt.show()

# Feature Importance from Random Forest (Best for interpretation)
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x=feat_imp, y=feat_imp.index)
plt.title('Feature Importance - Random Forest')
plt.show()

# Actual vs Predicted plot for best model
best_model_name = metrics_df.sort_values(by='R2', ascending=False).iloc[0]['Model']
best_pred = results[best_model_name]

plt.figure(figsize=(6,6))
plt.scatter(y_test, best_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Actual vs Predicted - {best_model_name}')
plt.show()

# ============== CONCLUSION FOR REPORT ==============
print("""
CONCLUSION:

Best-performing: Gradient Boosting / Random Forest (Usually R2 ~0.80)
- Justification: Ensemble methods capture non-linear relationships between location, 
  income, and house age that Linear Regression misses. Gradient Boosting corrects 
  errors iteratively, giving highest R2 and lowest MSE.

Worst-performing: Linear Regression / SVR
- Linear Regression Reason: Housing data is highly non-linear (e.g., Latitude/Longitude effect is not linear).
- SVR Reason: Very slow on 20k rows and RBF kernel with default hyperparameters 
  underfits without extensive tuning. Also sensitive to noise.

If your run shows SVR as worst -> State that. If Linear Regression is worst -> State that.
Both are acceptable with proper reasoning.
""")
