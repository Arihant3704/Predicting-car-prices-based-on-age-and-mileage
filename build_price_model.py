import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid Qt plugin issues
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV data
df = pd.read_csv('Car_details_v3.csv')

# Plot original data distributions
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
sns.histplot(df['year'].dropna(), bins=30, kde=True)
plt.title('Year Distribution (Original)')

plt.subplot(1, 3, 2)
sns.histplot(df['km_driven'].dropna(), bins=30, kde=True)
plt.title('KM Driven Distribution (Original)')

plt.subplot(1, 3, 3)
sns.histplot(df['selling_price'].dropna(), bins=30, kde=True)
plt.title('Selling Price Distribution (Original)')

plt.tight_layout()
plt.savefig('original_data_distributions.png')
plt.show()

# Preprocess data: calculate age from year
current_year = datetime.now().year
df['age'] = current_year - df['year']

# Plot age distribution after preprocessing
plt.figure(figsize=(6, 4))
sns.histplot(df['age'].dropna(), bins=30, kde=True)
plt.title('Age Distribution (Preprocessed)')
plt.savefig('age_distribution.png')
plt.show()

# Use only relevant columns and drop rows with missing values in these columns
df_model = df[['age', 'km_driven', 'selling_price']].dropna()

# Filter out unrealistic data (e.g., negative or zero prices, negative age or km)
df_model = df_model[(df_model['selling_price'] > 0) & (df_model['age'] >= 0) & (df_model['km_driven'] >= 0)]

# Plot correlation heatmap
plt.figure(figsize=(6, 4))
corr = df_model.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.show()

# Scatter plots of features vs target
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x='age', y='selling_price', data=df_model, alpha=0.5)
plt.title('Age vs Selling Price')

plt.subplot(1, 2, 2)
sns.scatterplot(x='km_driven', y='selling_price', data=df_model, alpha=0.5)
plt.title('KM Driven vs Selling Price')

plt.tight_layout()
plt.savefig('feature_vs_price_scatter.png')
plt.show()

# Prepare features and target
X = df_model[['age', 'km_driven']]
y = df_model['selling_price']

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict on training data for residual plot
y_pred = model.predict(X)
residuals = y - y_pred

# Residual plot
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.savefig('residual_plot.png')
plt.show()

# Save model parameters to JSON
model_params = {
    'intercept': model.intercept_,
    'coefficients': model.coef_.tolist(),
    'features': ['age', 'km_driven']
}

with open('model_params.json', 'w') as f:
    json.dump(model_params, f, indent=4)

print("Model training complete. Parameters saved to 'model_params.json'.")
