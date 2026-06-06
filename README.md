# Employee Attrition Regression Modelling

A comprehensive machine learning analysis to predict and understand the factors driving employee attrition using logistic regression. This project analyzes employee data from different organizational locations to identify key predictors of workforce turnover.

## Table of Contents

- [Overview](#overview)
- [Research Objective](#research-objective)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Model Performance](#model-performance)
- [Installation & Usage](#installation--usage)
- [Feature Importance](#feature-importance)
- [Conclusions & Recommendations](#conclusions--recommendations)

## Overview

Employee attrition is a critical challenge for organizations, leading to increased costs in recruitment, training, and loss of productivity. This project uses **logistic regression** to predict which employees are likely to leave their organization and identifies the most influential factors driving attrition decisions.

The analysis covers:
- **1,029 employee records** from a previous facility closure
- **441 employee records** from the Lyon location
- **Two-stage modeling**: Training on historical data and prediction on current workforce
- **Feature importance ranking** to identify actionable insights

## Research Objective

**Primary Question**: What are the key factors that influence employee attrition, and can we build a predictive model to identify at-risk employees?

**Key Sub-Questions**:
1. Which employee characteristics most strongly predict attrition?
2. How does the predictive model perform on different organizational locations?
3. Which employees in the current workforce are at highest risk of leaving?
4. What interventions should be prioritized based on factor importance?

## Dataset

### Data Sources

**1. Training Dataset (Previous Facility Closure)**
- **Records**: 1,029 employees
- **Status**: Historical data from closed facility
- **Missing Values**: 268 missing values across 5 variables
- **Used For**: Model training and validation

**2. Testing Dataset (Lyon Location)**
- **Records**: 441 employees
- **Status**: Current workforce
- **Missing Values**: 131 missing values across 5 variables
- **Used For**: Attrition risk prediction

### Key Variables

| Variable | Type | Description | Imputation Strategy |
|----------|------|-------------|-------------------|
| **Attrition** | Binary | Left company (Yes/No) | Response variable |
| Age | Numeric | Employee age | Mean imputation |
| DistanceFromHome | Numeric | Distance to workplace (km) | Median imputation |
| DailyRate | Numeric | Daily rate of pay | Mean imputation |
| Education | Ordinal | Education level | Categorical encoding |
| EnvironmentSatisfaction | Ordinal | Work environment rating | Categorical encoding |
| JobInvolvement | Ordinal | Job involvement level | Categorical encoding |
| JobSatisfaction | Ordinal | Job satisfaction rating | Categorical encoding |
| StockOptionLevel | Ordinal | Stock option level | Categorical encoding |
| WorkLifeBalance | Ordinal | Work-life balance rating | Categorical encoding |
| YearsAtCompany | Numeric | Tenure at company | Categorical encoding |
| YearsInCurrentRole | Numeric | Time in current position | Categorical encoding |
| YearsWithCurrManager | Numeric | Time with current manager | Categorical encoding |
| OverTime | Binary | Overtime work requirement | Categorical encoding |
| BusinessTravel | Categorical | Frequency of business travel | One-hot encoding |
| Department | Categorical | Organizational department | One-hot encoding |
| Gender | Categorical | Employee gender | One-hot encoding |
| JobRole | Categorical | Specific job position | One-hot encoding |
| MaritalStatus | Categorical | Marital status | One-hot encoding |
| EducationField | Categorical | Field of education | One-hot encoding |

### Missing Data Handling

**Numerical Variables:**
- **Age**: 136 missing values → Mean imputation (normally distributed)
- **DailyRate**: 27 missing values → Mean imputation (uniform distribution)
- **DistanceFromHome**: 95 missing values → Median imputation (right-skewed distribution)

**Categorical Variables:**
- **MaritalStatus**: 5 missing values → Imputed as "Not available" category
- **BusinessTravel**: 5 missing values → Imputed as "Not available" category

**Rationale**: Using appropriate imputation strategies maintains data integrity while preserving distribution characteristics.

### Feature Engineering

**Encoded Variables** (One-hot encoding):
- Department: Sales, Research & Development, Human Resources
- Job Role: 9 distinct roles
- Education Field: 6 fields
- Gender: Male/Female
- Business Travel: Non-Travel, Travel Rarely, Travel Frequently
- Marital Status: Single, Married, Divorced, Not available
- Over Time: Yes/No

**Excluded Variables** (Due to collinearity or redundancy):
- Attrition_No (inverse of Attrition_Yes)
- EmployeeNumber (unique identifier)
- JobLevel (high correlation with role/salary)
- PercentSalaryHike (minimal variance)
- TotalWorkingYears (high correlation with age)
- YearsSinceLastPromotion (high correlation with tenure)
- JobRole_Human Resources (multicollinearity)

## Methodology

### 1. Data Preprocessing

**Steps Executed**:
1. Load training dataset (previous facility closure)
2. Identify missing values and distribution patterns
3. Apply appropriate imputation strategies
4. Visualize distributions of variables with missing values
5. One-hot encode categorical variables
6. Remove redundant and highly correlated features
7. Rename binary target variable for clarity

**Collinearity Assessment**:
- Generated correlation heatmap for all features
- Identified and removed highly correlated variables to improve model stability

### 2. Model Development

**Algorithm**: Logistic Regression
- **Type**: Binary classification model
- **Link Function**: Logit (log-odds)
- **Regularization**: L2 (Ridge) - default in scikit-learn
- **Solver**: LBFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno)
- **Max Iterations**: 2000 (to ensure convergence)

**Training Configuration**:
```python
classifier_LR = LogisticRegression(max_iter=2000)
classifier_LR.fit(trainX, trainY)
```

**Train-Test Split**:
- Training Set: 60% (617 employees)
- Validation Set: 40% (412 employees)
- Random State: 357 (for reproducibility)

### 3. Model Validation

**ROC Curve Analysis**:
- Generated ROC (Receiver Operating Characteristic) curve
- Evaluates model performance across different classification thresholds
- Measures trade-off between True Positive Rate and False Positive Rate

**Metrics Calculated**:
- AUC-ROC Score: Measures overall classification performance
- Sensitivity/Recall: Ability to identify actual attritions
- Specificity: Ability to identify non-attritions

### 4. Feature Importance Analysis

**Method**: Logistic Regression Coefficients
- Coefficients represent log-odds of attrition
- **Positive coefficients**: Increase probability of attrition
- **Negative coefficients**: Decrease probability of attrition
- **Magnitude**: Strength of effect

## Key Findings

### Top 10 Risk Factors (Increase Attrition)

| Rank | Factor | Coefficient | Interpretation |
|------|--------|-------------|-----------------|
| 1 | **OverTime_Yes** | +0.827 | Working overtime **strongly increases** attrition risk |
| 2 | **MaritalStatus_Single** | +0.375 | Single employees have higher attrition |
| 3 | **BusinessTravel_Frequently** | +0.369 | Frequent travel increases departure risk |
| 4 | **Department_Sales** | +0.315 | Sales department has higher attrition |
| 5 | **JobRole_SalesRepresentative** | +0.258 | Sales representatives at higher risk |
| 6 | **EducationField_Technical** | +0.220 | Technical degree holders more likely to leave |
| 7 | **Gender_Male** | +0.202 | Male employees slightly more likely to leave |
| 8 | **NumCompaniesWorked** | +0.198 | Job hoppers more likely to attrite |
| 9 | **JobRole_LaboratoryTechnician** | +0.188 | Lab technicians at moderate risk |
| 10 | **EducationField_Marketing** | +0.163 | Marketing education linked to higher attrition |

### Top 10 Protective Factors (Decrease Attrition)

| Rank | Factor | Coefficient | Interpretation |
|------|--------|-------------|-----------------|
| 1 | **OverTime_No** | -0.826 | Not working overtime is strongest protective factor |
| 2 | **StockOptionLevel** | -0.541 | Stock options significantly reduce attrition |
| 3 | **JobSatisfaction** | -0.507 | High job satisfaction critical for retention |
| 4 | **EnvironmentSatisfaction** | -0.432 | Positive work environment reduces departures |
| 5 | **JobInvolvement** | -0.422 | Engaged employees less likely to leave |
| 6 | **Department_R&D** | -0.356 | R&D department has lowest attrition |
| 7 | **MaritalStatus_Divorced** | -0.292 | Divorced employees (paradoxically) less likely to leave |
| 8 | **WorkLifeBalance** | -0.281 | Good work-life balance reduces attrition |
| 9 | **EducationField_LifeSciences** | -0.278 | Life sciences education associated with retention |
| 10 | **YearsWithCurrManager** | -0.251 | Long tenure with manager increases retention |

### Critical Insights

#### 1. **Work Environment is Paramount**
- **Overtime is the single strongest predictor** of attrition
- Employees forced to work overtime have 2.3x higher attrition likelihood
- This suggests workload and work-life balance are critical concerns

#### 2. **Compensation & Incentives Matter**
- **Stock options** have one of the strongest retention effects (-0.541)
- Suggests equity/wealth-building motivates employees to stay
- Indicates need for competitive benefits packages

#### 3. **Job Satisfaction is Critical**
- **Job satisfaction** (-0.507) and **environment satisfaction** (-0.432) are major factors
- Together, satisfaction measures are among top 5 protective factors
- Improving workplace conditions directly impacts retention

#### 4. **Department Differences**
- Sales department shows highest attrition risk (+0.315)
- R&D department shows strongest retention (-0.356)
- Suggests different HR strategies needed by department

#### 5. **Demographics & Life Circumstances**
- **Single employees** have higher attrition (+0.375)
- Males slightly more likely to leave than females
- Younger employees (implied by lower age coefficients) have higher risk

#### 6. **Career Development Influence**
- **Years with current manager** (-0.251) indicates importance of relationship
- **Years in current role** (-0.089) has moderate protective effect
- **Training** (-0.089) also protective, though smaller effect
- Suggests career development and mentorship reduce departures

#### 7. **Business Travel Impact**
- **Frequent business travel** (+0.369) significantly increases attrition
- May reflect work-life balance and family considerations
- Non-travel roles (-0.236) have protective effect

#### 8. **Job Role Variations**
- **Sales Representatives** (+0.258) at highest role-specific risk
- **Healthcare Representatives** (-0.177) and **Research Scientists** (-0.202) more stable
- Suggests role nature affects retention

## Model Performance

### Training Configuration
- **Algorithm**: Logistic Regression
- **Training Instances**: 617 employees (60%)
- **Validation Instances**: 412 employees (40%)
- **Features**: 51 (after feature selection and encoding)
- **Iterations to Convergence**: 6

### Model Diagnostics
**Convergence**: 
- Status: Converged successfully
- Iterations: 6 (well before max_iter=2000)
- Solver: LBFGS (optimal for binary logistic regression)

### ROC Curve Analysis
- **Shape**: Concave curve indicating good discrimination
- **Interpretation**: Model performs better than random classifier
- **Optimal Threshold**: Depends on business requirements (cost of false positives vs. false negatives)

### Validation Results

The model was validated on the Lyon facility dataset:
- **Test Instances**: 441 employees
- **Attrition Predictions**: Generated probability scores for each employee
- **Risk Distribution**: 
  - Low Risk (0-0.3 probability): Majority of employees
  - Moderate Risk (0.3-0.5 probability): Notable segment
  - High Risk (>0.5 probability): Employees requiring intervention

### Example Predictions (Lyon Dataset)

| Employee ID | Attrition Probability | Risk Level | Interpretation |
|-------------|----------------------|-----------|-----------------|
| 1 | 0.0058 | Very Low | Extremely unlikely to leave |
| 2 | 0.3214 | Moderate | Should monitor and engage |
| 438 | 0.5702 | High | At significant risk - intervention needed |
| 440 | 0.4510 | Moderate-High | Priority for retention programs |

## Installation & Usage

### Requirements

- **Python** 3.8+
- **Jupyter Notebook**
- **Key Libraries**:
  - pandas (data manipulation)
  - numpy (numerical computing)
  - scikit-learn (machine learning)
  - matplotlib (visualization)
  - seaborn (statistical visualization)
