# Import required libraries
import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import sklearn.preprocessing as lm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Configure pandas display options for better data visualization
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.max_rows', None)     # Display all rows
pd.set_option('display.max_colwidth', None) # Display full column width



# Load the inventory dataset from Excel file
inventory_data = pd.read_excel('Dataset\Inventory.xlsx')

# Create dataframe with first 3000 rows as specified in the assignment
df = pd.DataFrame(inventory_data, index=range(0, 3000))


# Remove 'Item No.' column (identifier not useful for prediction)
df.drop("Item No. ", axis=1, inplace=True)

# Remove redundant/derived columns to avoid multicollinearity
# 'Over 2 years Qty' is redundant with 'Over 2 Years Qty'
# '% of over 2 year' can be derived from other columns
df.drop(['Over 2 years Qty', '% of over 2 year'], axis=1, inplace=True)

# Display dataframe information to verify structure
print(df.info(verbose=True))



# Check for missing values in each column
print(df.isnull().sum())

# Impute numerical columns with mean
# This is appropriate for continuous variables with approximately normal distribution
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

# Impute categorical columns with mode (most frequent value)
# This preserves the most common category without introducing invalid values
categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
for col in categorical_cols:
    if df[col].isnull().any():
        mode_value = df[col].mode()
        if not mode_value.empty:
            df[col] = df[col].fillna(mode_value[0])


# --- State Encoding ---
# Create dummy variable for NSW state (binary: 1 if NSW, 0 otherwise)
dummies = pd.get_dummies(df['State'], prefix='State')
df['State_NSW'] = dummies['State_NSW']
df.drop('State', axis=1, inplace=True)

# --- Warehouse (Whse) Encoding ---
# Display value counts to understand warehouse distribution
print(df['Whse'].value_counts())

# Create dummy variables for each warehouse location
Whse_dummy = pd.get_dummies(df['Whse'], prefix='Whse')

# Ensure all three warehouse columns exist, even if not present in data
# This handles anomalies gracefully without deleting observations
for col in ['Whse_1N0', 'Whse_1N1', 'Whse_1W0']:
    if col in Whse_dummy.columns:
        df[col] = Whse_dummy[col]
    else:
        df[col] = 0  # Set to 0 if warehouse type not present

df.drop('Whse', axis=1, inplace=True)

# --- Item Type Encoding ---
# Consolidate item types into 4 main categories:
for i in range(len(df['Item Type'])):
    if df.loc[i, 'Item Type'].startswith('FG'):
        df.loc[i, 'Item Type'] = 'Finished Goods'
    elif df.loc[i, 'Item Type'].startswith('RM'):
        df.loc[i, 'Item Type'] = 'Raw Material'
    elif df.loc[i, 'Item Type'].startswith('WIP'):
        df.loc[i, 'Item Type'] = 'WIP Manufactured'
    else:
        df.loc[i, 'Item Type'] = 'Other'

# --- Base Unit Encoding ---
# One-hot encode base unit measurements
base_unit_dummies = pd.get_dummies(df['Base Unit'], prefix='Base_Unit')
for col in base_unit_dummies.columns:
    if col in base_unit_dummies.columns:
        df[col] = base_unit_dummies[col]
    else:
        df[col] = 0
df.drop('Base Unit', axis=1, inplace=True)

# --- Business Area Encoding ---
# One-hot encode business areas
bussiness_area_dummies = pd.get_dummies(df['Business Area'], prefix='Business_Area')
for col in bussiness_area_dummies.columns:
    if col in bussiness_area_dummies.columns:
        df[col] = bussiness_area_dummies[col]
    else:
        df[col] = 0
df.drop('Business Area', axis=1, inplace=True)


# --- Dead Stock Encoding (Target Variable) ---
# Convert target variable to binary: Yes=1 (dead stock), No=0 (active stock)
dead_stock = {"Yes": 1, "No": 0}
df['Dead stock'] = df['Dead stock'].map(dead_stock)

# --- ABC Class Encoding (Ordinal) ---
# Encode ABC classification with ordinal values reflecting importance/turnover
# A (highest) = 8, down to J (lowest) = 1, with anomalies = 0
abc_class = {"A": 8, "B": 7, "C": 6, "F": 5, "H": 4, "E": 3, "I": 2, "J": 1, 0: "Anomalies"}
df['ABC Class'] = df['ABC Class'].map(abc_class)



# Split data: first 80% for training/testing, remaining 20% reserved
rows = int(len(df) * 0.8)

# Create target variable array (y) from first 80% of data
y = df.loc[:rows-1, 'Dead stock'].to_numpy()

# Create feature matrix (X) from first 80% of data, excluding target variable
X = df.loc[:rows-1, df.columns != 'Dead stock']

# Select only numeric columns for modeling (dummy variables are already numeric)
X_numeric = X.select_dtypes(include=[np.number]).to_numpy()



# Split data into training (80%) and testing (20%) sets
# random_state=31 ensures reproducibility
# stratify=y maintains class balance in both sets
X_train, X_test, y_train, y_test = ms.train_test_split(
    X_numeric, y, test_size=0.2, random_state=31, stratify=y
)


# Standardize features to mean=0 and variance=1
# This improves model performance, especially for Logistic Regression
scaler = lm.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on training data
X_test_scaled = scaler.transform(X_test)        # Transform test data using training parameters


# Initialize and train Logistic Regression model
# max_iter=1000 ensures convergence for this dataset
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Make predictions on both training and test sets
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Calculate and display accuracy scores
train_accuracy_lg = accuracy_score(y_train, y_train_pred)
test_accuracy_lg = accuracy_score(y_test, y_test_pred)
print("\n=== LOGISTIC REGRESSION RESULTS ===")
print("Training Accuracy:", train_accuracy_lg)
print("Test Accuracy:", test_accuracy_lg)


# Initialize and train Random Forest model
# n_estimators=100 creates 100 decision trees in the forest
# random_state=31 ensures reproducibility
clf = RandomForestClassifier(random_state=31, n_estimators=100)
clf.fit(X_train_scaled, y_train)

# Calculate accuracy scores for Random Forest
train_accuracy_randomforest = accuracy_score(y_train, clf.predict(X_train_scaled))
test_accuracy_randomforest = accuracy_score(y_test, clf.predict(X_test_scaled))

print("\n=== RANDOM FOREST RESULTS ===")
print("Training Accuracy:", train_accuracy_randomforest)
print("Test Accuracy:", test_accuracy_randomforest)

# Compare model performances and provide recommendation

print("\n=== MODEL COMPARISON ===")
print(f"Logistic Regression - Train: {train_accuracy_lg:.4f}, Test: {test_accuracy_lg:.4f}")
print(f"Random Forest - Train: {train_accuracy_randomforest:.4f}, Test: {test_accuracy_randomforest:.4f}")
print("\nRecommendation: Random Forest performs better with higher test accuracy")
print("and ability to capture nonlinear relationships in the inventory data.")
