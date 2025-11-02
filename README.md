# Dead Stock Prediction - Inventory Analysis

A machine learning project to predict dead stock items in warehouse inventory using classification algorithms.

## 📋 Project Overview

This project analyses inventory data from a mid-sized manufacturing retailer with warehouses across Australia to identify items likely to become dead stock (i.e., those with no recent sales activity, no longer in marketing campaigns, and limited relevance to current customer needs).

### Business Problem

The company noticed certain items had not moved for several months. The inventory manager needed to:
- Classify items as dead stock or active
- Make informed decisions about markdowns, liquidation, or disposal
- Free up valuable warehouse space
- Enhance the precision of warehouse facility storage planning

## 🎯 Objective

Predict which stock items are likely to be dead stock using various data mining methods, specifically comparing linear and nonlinear classification algorithms.

**Target Variable:** Dead stock (1=Yes, 0=No)

## 📊 Dataset

The dataset contains 3,000 inventory records with 30+ features, including:

### Key Features:
- **Location Data:** Warehouse (Whse), State
- **Item Information:** Item Description, Item Type, Base Unit, Business Area
- **Quantity Metrics:** Total Quantity, 6 Months QTY, 12 Months QTY, 2 Years QTY, Over 2 Years Qty, Over 3 Years Quantity
- **Financial Data:** Inventory Ageing Report, Unit Cost, Total Value
- **Monthly Usage:** Jan through Dec monthly quantities
- **Performance Indicators:** Average monthly usage, Inventory Turn, ABC Class

### Variable Types:
- **Numeric (22):** Quantities, costs, values, monthly usage metrics
- **Ordinal (2):** ABC Class, Dead stock
- **Nominal (6):** Item Description, Warehouse, State, Base Unit, Business Area, Item Type

## 🛠️ Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
  - Logistic Regression (Linear Classifier)
  - Random Forest Classifier (Nonlinear Classifier)
  - StandardScaler for feature scaling
  - Train-test split with stratification

## 📁 Project Structure

```
├── Inventory.xlsx          # Raw inventory data
├── dead_stock_prediction.py # Main prediction script
├── requirements.txt #requirements to run the script    
└── README.md                    # Project documentation
```


## 💻 Usage

The script will:
1. Load and preprocess the inventory data
2. Handle missing values
3. Encode categorical variables
4. Split data into training (80%) and testing (20%) sets
5. Train both Logistic Regression and Random Forest models
6. Output training and testing accuracies

## 🔍 Methodology

### Data Preprocessing
1. **Missing Value Imputation:**
   - Numerical features: Mean imputation
   - Categorical features: Mode imputation

2. **Feature Engineering:**
   - State: Converted to dummy variable (State_NSW)
   - Warehouse (Whse): One-hot encoded (1N0, 1N1, 1W0)
   - Item Type: Consolidated into 4 categories (Finished Goods, Raw Materials, WIP Manufactured, Other)
   - Base Unit: One-hot encoded
   - Business Area: One-hot encoded
   - ABC Class: Ordinal encoding (A=8, B=7, C=6, F=5, H=4, E=3, I=2, J=1)
   - Dead stock: Binary encoding (Yes=1, No=0)

3. **Feature Scaling:**
   - StandardScaler applied to normalise features (mean=0, variance=1)

### Model Training
- **80/20 Train-Test Split** with stratification
- **Random State:** 31 (for reproducibility)

## 📈 Results

### Model Performance

| Model | Training Accuracy | Test Accuracy |
|-------|------------------|---------------|
| Logistic Regression | 94.43% | 94.38% |
| Random Forest | 100.00% | 95.00% |

### Key Findings
- **Random Forest** outperforms Logistic Regression with better test accuracy
- The model successfully captures complex nonlinear relationships in inventory data
- Slight overfitting in Random Forest (100% training accuracy), but still generalises well

### Most Important Features
1. **Over 2 Years Qty** - Strongest indicator of dead stock
2. **Inventory Turn** - Measures stock movement velocity
3. **Total Quantity** - Provides inventory scale context

## 🎓 Insights & Recommendations

### Recommended Model: **Random Forest Classifier**

**Rationale:**
- Superior handling of nonlinear relationships and feature interactions
- Higher accuracy on test data (95% vs 94.38%)
- Better at managing anomalies and complex patterns
- Despite slight overfitting, it demonstrates strong generalisation

### Business Implications
1. Items with high "Over 2 Years Qty" should be prioritised for review
2. Low inventory turnover is a critical warning sign
3. ABC classification combined with age metrics provides powerful predictive signals
4. Warehouse location (especially 1N1) shows different dead stock patterns


## 🤝 Contributing

This is an academic project. For questions or suggestions, please open an issue.

## 📄 License

This project is created for educational purposes.

## 👤 Author

Nhat Nam (Tom) Nguyen
