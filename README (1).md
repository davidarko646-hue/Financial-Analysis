# Loan Default Prediction Using Ensemble Methods

**Author:** David Arko  
**Date:** February 2026

## Project Overview

This project demonstrates the power of ensemble learning methods for predicting loan defaults. By combining 5 different machine learning algorithms, we achieved **100% accuracy** on test data.

### Business Problem

Banks need to predict which loan applicants are likely to default to:
- Minimize financial risk
- Make informed lending decisions
- Protect shareholders and depositors
- Maintain financial stability

### Solution

Implemented an ensemble voting classifier that combines:
1. **Logistic Regression** - Linear baseline model
2. **Random Forest** - Tree-based ensemble for non-linear patterns
3. **Gradient Boosting** - Sequential learning from mistakes
4. **Support Vector Machine** - Complex decision boundaries
5. **Neural Network** - Deep learning for hidden patterns

## Dataset

**188 loan applications** with the following features:

| Feature | Description |
|---------|-------------|
| **age** | Applicant's age |
| **income** | Annual income (GHS) |
| **loan_amount** | Requested loan amount |
| **credit_score** | Credit score (300-850 scale) |
| **employment_length** | Years at current job |
| **debt_to_income** | Debt-to-income ratio |
| **previous_defaults** | History of defaults (0/1) |
| **loan_purpose** | Purpose: business, home, car, education, personal |
| **default** | Target variable: 0 = Paid, 1 = Defaulted |

**Class Distribution:**
- Non-Defaulters: 150 (80%)
- Defaulters: 38 (20%)

## Results

### Individual Model Performance

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 92.11% |
| Random Forest | **100.00%** |
| Gradient Boosting | **100.00%** |
| SVM | 92.11% |
| Neural Network | **100.00%** |
| **ENSEMBLE (Voting)** | **100.00%** |

### Key Findings

**Top Predictive Features (by importance):**
1. Previous Defaults (0.88 correlation)
2. Debt-to-Income Ratio (0.80 correlation)
3. Credit Score (-0.78 correlation)
4. Age (-0.67 correlation)
5. Income (-0.62 correlation)

**High-Risk Profile:**
- Credit score < 600
- Personal or education loans
- Previous default history
- High debt-to-income ratio (>0.45)
- Low income (<GHS 35,000)

**Low-Risk Profile:**
- Credit score > 650
- Business, home, or car loans
- No previous defaults
- Low debt-to-income ratio (<0.35)
- Higher income (>GHS 60,000)

## Technical Details

### Technologies Used
- **Python 3.x**
- **scikit-learn** - Machine learning models
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **matplotlib** - Data visualization

### Methodology

**1. Data Preprocessing:**
- Encoded categorical variables (loan purpose)
- Scaled features for distance-based algorithms
- Split data: 80% training, 20% testing
- Stratified sampling to maintain class balance

**2. Model Training:**
- Trained 5 independent models
- Each model learns different patterns
- Logistic Regression & SVM use scaled data
- Tree-based models use raw data

**3. Ensemble Creation:**
- Voting Classifier (Majority Voting)
- Each model gets one vote
- Final prediction = majority consensus
- Reduces individual model biases

**4. Evaluation:**
- Accuracy score
- Precision, Recall, F1-score
- Confusion matrix analysis
- Feature importance ranking

## Visualizations

### Model Comparison
![Model Comparison](model_comparison.png)

### Feature Importance
![Feature Importance](feature_importance.png)

## How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib
```

### Execution
```bash
python ensemble_loan_prediction.py
```

### Expected Output
- Model training progress
- Individual model accuracies
- Ensemble performance metrics
- Classification report
- Confusion matrix
- Feature importance rankings
- Saved visualizations

## Project Structure

```
loan-default-ensemble/
│
├── loan_default_data.csv          # Dataset
├── ensemble_loan_prediction.py    # Main script
├── model_comparison.png           # Results visualization
├── feature_importance.png         # Feature analysis
└── README.md                      # This file
```

## Key Insights for Banking

**Policy Recommendations:**

1. **Automatic Rejection Criteria:**
   - Credit score < 600
   - Any previous defaults
   - Debt-to-income ratio > 0.50

2. **High Scrutiny Required:**
   - Personal loans (97% default rate in data)
   - Applicants aged 20-30
   - Income < GHS 35,000

3. **Low Risk Approvals:**
   - Business, home, or car loans
   - Credit score > 650
   - No previous defaults
   - DTI < 0.35

4. **Ensemble Benefits:**
   - More robust than single model
   - Reduces false negatives (missed defaults)
   - Combines strengths of different algorithms

## 📝 Lessons Learned

**Why Ensemble Works:**

1. **Diversity:** Different models catch different patterns
2. **Error Correction:** When one model fails, others compensate
3. **Reduced Overfitting:** Averaging reduces model-specific biases
4. **Increased Confidence:** Multiple models agreeing = higher reliability

**Real-World Application:**

This approach is used by major financial institutions worldwide:
- Credit scoring agencies
- Banks' risk management departments
- Fintech companies
- Insurance underwriting

## Educational Value

**Topics Demonstrated:**
- Supervised learning (classification)
- Ensemble methods (voting)
- Model comparison and evaluation
- Feature engineering
- Data preprocessing
- Imbalanced dataset handling
- Business problem solving with ML

## About the Author

**David Arko**  
Data Science Professional

Specializing in:
- Machine Learning
- Statistical Analysis
- Financial Risk Modeling
- Economic Forecasting

**Current Role:** Data Analyst at Bank of Ghana (Starting May 2026)

## License

This project is available for educational and portfolio purposes.

---

**Note:** This is a demonstration project using synthetic data for educational purposes. Real-world deployment would require extensive validation, regulatory compliance, and continuous monitoring.

## Acknowledgments

- scikit-learn documentation and community
- Banking domain knowledge from industry research

---

*"The best predictor of future behavior is past behavior."* - Applied in this project through the 'previous_defaults' feature, which showed 88% correlation with loan default.
