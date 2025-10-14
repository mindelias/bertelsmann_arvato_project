# Customer Segmentation and Response Prediction for Arvato Financial Services

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning project that uses demographic data to identify customer segments and predict response rates for marketing campaigns. This is the capstone project for the Udacity Data Science Nanodegree, completed in partnership with Bertelsmann Arvato Analytics.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Results Summary](#results-summary)
- [Installation](#installation)
- [Data](#data)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## 🎯 Project Overview

This project addresses a real-world business challenge: **How can a mail-order company efficiently identify potential customers from the general population to improve marketing ROI?**

The solution combines:
1. **Unsupervised Learning** (Part 1): Customer segmentation using PCA and K-Means clustering
2. **Supervised Learning** (Part 2): Response prediction using ensemble classification models
3. **Business Insights** (Part 3): Actionable recommendations for targeted marketing

### Business Context

Traditional direct marketing campaigns achieve only 1-2% response rates, meaning 98-99% of marketing spend is wasted. This project demonstrates how data science can:
- Identify high-value customer segments
- Predict campaign responses with 75%+ accuracy (ROC-AUC > 0.75)
- Reduce marketing costs by 40-50% while maintaining customer acquisition

## 💡 Problem Statement

**Challenge:** A German mail-order company needs to optimize their customer acquisition strategy by targeting individuals most likely to respond to marketing campaigns.

**Approach:** 
- Analyze 891,221 individuals from the general population
- Compare with 191,652 existing customers
- Build predictive models on 42,982 campaign recipients
- Generate predictions for 42,833 test individuals

**Success Metrics:**
- ROC-AUC Score ≥ 0.70 (achieved: **0.97**)
- 2-3x improvement in response rate through targeting (achieved: **3.65x** for top cluster)
- Clear, actionable customer segments identified

## 🏆 Results Summary

### Part 1: Customer Segmentation
- **Dimensionality Reduction:** 366 features → 85 PCA components (85% variance retained)
- **Optimal Clusters:** K = 14 segments identified
- **Top Customer Segments:**
  - **Cluster 0:** 3.65x over-represented (PRIMARY TARGET) 🎯
  - **Cluster 7:** 2.21x over-represented
  - **Cluster 13:** 2.09x over-represented
- **Avoid Segments:** Clusters 2, 8, 12 (under-represented)

### Part 2: Response Prediction
- **Best Model:** Gradient Boosting Classifier
- **ROC-AUC Score:** 0.9709 (validation set) 🔥
- **Top Predictive Features:**
  1. Cluster_11 (most important)
  2. Cluster membership (raw feature)
  3. Cluster_13
  4. PC64, PC44, PC31 (PCA components)
- **Class Imbalance Handling:** SMOTE + Random Undersampling

### Key Insight
**Cluster membership features from Part 1 were the most important predictors in Part 2**, validating the two-stage approach and demonstrating how unsupervised learning insights directly improved supervised model performance.

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook
- 8GB+ RAM recommended (for large dataset processing)

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/bertelsmann_arvato_project.git
cd bertelsmann_arvato_project
```

2. **Create a virtual environment (recommended):**
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n arvato python=3.8
conda activate arvato
```

3. **Install required packages:**
```bash
pip install -r requirements.txt
```

## 📊 Data

### Dataset Overview

The project uses four datasets provided by Bertelsmann Arvato Analytics:

| Dataset | Rows | Columns | Description |
|---------|------|---------|-------------|
| `Udacity_AZDIAS_052018.csv` | 891,221 | 366 | General German population demographics |
| `Udacity_CUSTOMERS_052018.csv` | 191,652 | 369 | Existing customer demographics |
| `Udacity_MAILOUT_052018_TRAIN.csv` | 42,982 | 367 | Campaign training data (with RESPONSE labels) |
| `Udacity_MAILOUT_052018_TEST.csv` | 42,833 | 366 | Campaign test data (labels withheld) |

**Additional Files:**
- `DIAS Attributes - Values 2017.xlsx`: Feature value mappings
- `DIAS Information Levels - Attributes 2017.xlsx`: Feature descriptions

### Getting the Data

**⚠️ Important:** The data files are NOT included in this repository due to size and licensing restrictions.

**To obtain the data:**

1. **Udacity Students:** Download from the Udacity workspace
   - Navigate to your capstone project workspace
   - Download all four CSV files and two Excel files
   
2. **Non-Udacity Users:** This data is proprietary to Bertelsmann Arvato Analytics
   - Contact Arvato directly for licensing: https://www.arvato.com
   - Note: Academic use may require special permissions

3. **Place data files:**
```
bertelsmann_arvato_project/
├── data/
│   ├── Udacity_AZDIAS_052018.csv
│   ├── Udacity_CUSTOMERS_052018.csv
│   ├── Udacity_MAILOUT_052018_TRAIN.csv
│   ├── Udacity_MAILOUT_052018_TEST.csv
│   ├── DIAS Attributes - Values 2017.xlsx
│   └── DIAS Information Levels - Attributes 2017.xlsx
```

### Data Characteristics

**Features (366 total):**
- **Person:** Age, gender, income, education
- **Household:** Structure, composition, children
- **Building:** Type, construction year
- **Neighborhood:** Economic status, urbanization
- **Transaction (D19):** Purchase behavior across categories
- **Behavioral (SEMIO):** Lifestyle typologies
- **Vehicle (KBA):** Car ownership and preferences

**Challenges:**
- High dimensionality (366 features)
- Missing values encoded as -1, 0, 9, 'X'
- Class imbalance (1-2% response rate)
- Mixed numeric and categorical types

## 📁 Project Structure
```
bertelsmann_arvato_project/
├── data/                                    # [IN .GITIGNORE]
│   ├── Udacity_AZDIAS_052018.csv           # General population (raw)
│   ├── Udacity_CUSTOMERS_052018.csv        # Customers (raw)
│   ├── Udacity_MAILOUT_052018_TRAIN.csv   # Training data (raw)
│   ├── Udacity_MAILOUT_052018_TEST.csv    # Test data (raw)
│   ├── azdias_cleaned.csv                  # Preprocessed population
│   ├── azdias_cleaned.pkl                  # Preprocessed (pickle - faster)
│   ├── customers_cleaned.csv               # Preprocessed customers
│   ├── customers_cleaned.pkl               # Preprocessed (pickle)
│   ├── preprocessing_metadata.json         # Cleaning metadata
│   ├── DIAS Attributes - Values 2017.xlsx  # Feature value mappings
│   └── DIAS Information Levels - Attributes 2017.xlsx  # Feature descriptions
├── models/                                  # [Create this folder]
│   ├── scaler.pkl                          # Fitted StandardScaler
│   ├── pca.pkl                             # Fitted PCA transformer
│   ├── kmeans.pkl                          # Fitted K-Means model
│   └── best_model.pkl                      # Best classification model
├── Arvato Project Workbook.ipynb           # Main Jupyter notebook
├── proposal.pdf                             # Project proposal document
├── cluster_comparison_analysis.png         # Visualization: Cluster comparison
├── feature_importance.png                   # Visualization: Feature importance
├── kmeans_optimal_k.png                    # Visualization: Elbow method
├── missing_data_analysis.png               # Visualization: Missing data
├── model_evaluation.png                     # Visualization: Model performance
├── pca_variance_analysis.png               # Visualization: PCA variance
├── test_predictions_distribution.png       # Visualization: Test predictions
├── kaggle_submission.csv                   # Final predictions for Kaggle
├── requirements.txt                         # Python dependencies
├── README.md                                # This file
├── LICENSE                                  # MIT License
└── .gitignore                               # Git ignore rules
```

## 🔬 Methodology

### Part 0: Data Preprocessing
```python
# Key preprocessing steps
1. Convert missing codes (-1, 0, 9, 'X') to NaN
2. Drop columns with >80% missing (e.g., ALTER_KIND3, ALTER_KIND4)
3. Drop rows with >50% missing values
4. Remove categorical and constant columns
5. Impute remaining missing with median
6. Result: 891K → 791K rows, 366 → 319 features
```

**Preprocessing Pipeline:**
```
Raw Data (891K × 366) 
→ Missing Code Conversion 
→ Column Filtering (>80% missing dropped)
→ Row Filtering (>50% missing dropped)
→ Categorical Removal
→ Median Imputation
→ Clean Data (791K × 319)
```

### Part 1: Customer Segmentation (Unsupervised Learning)

**Step 1: Feature Scaling**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(azdias_clean)
```

**Step 2: Dimensionality Reduction**
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=85)  # Retains 85% variance
X_pca = pca.fit_transform(X_scaled)
```

**Step 3: Clustering**
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=14, random_state=42)
clusters = kmeans.fit_predict(X_pca)
```

**Step 4: Segment Analysis**
- Compare cluster distributions between AZDIAS and CUSTOMERS
- Calculate over/under-representation ratios
- Identify high-value target segments

### Part 2: Response Prediction (Supervised Learning)

**Step 1: Feature Engineering**
```python
# Leverage Part 1 insights
features = pd.DataFrame(X_pca, columns=[f'PC{i}' for i in range(85)])
features['Cluster'] = clusters
features['is_high_value_cluster'] = clusters.isin([0, 7, 13])
features = pd.concat([features, pd.get_dummies(clusters)], axis=1)
```

**Step 2: Handle Class Imbalance**
```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

smote = SMOTE(sampling_strategy=0.3, random_state=42)
undersampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)

X_resampled, y_resampled = smote.fit_resample(X, y)
X_balanced, y_balanced = undersampler.fit_resample(X_resampled, y_resampled)
```

**Step 3: Model Training**
```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)
```

**Step 4: Evaluation**
```python
from sklearn.metrics import roc_auc_score

y_pred_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
# Result: ROC-AUC = 0.9709
```

## 🔑 Key Findings

### 1. Customer Segment Profiles

**High-Value Segments (Over-represented):**

| Cluster | Ratio | % of Population | Characteristics |
|---------|-------|-----------------|-----------------|
| 0 | 3.65x | 9.5% | **PRIMARY TARGET** - Urban, eco-conscious, higher income |
| 7 | 2.21x | 6.0% | Suburban families, stable income |
| 13 | 2.09x | 5.7% | Young professionals, online shoppers |

**Low-Value Segments (Under-represented):**

| Cluster | Ratio | % of Population | Characteristics |
|---------|-------|-----------------|-----------------|
| 2 | 0.41x | 8.2% | **AVOID** - Rural, traditional, budget-conscious |
| 8 | 0.53x | 7.1% | Elderly, low mobility |
| 12 | 0.58x | 6.9% | Students, low income |

### 2. Feature Importance Analysis

**Top 15 Most Important Features:**
1. **Cluster_11** (0.070 importance) - Cluster dummy variable
2. **Cluster** (0.050 importance) - Raw cluster assignment
3. **Cluster_13** (0.045 importance) - High-value cluster flag
4. **PC64** (0.030 importance) - PCA component
5. **PC44** (0.029 importance) - PCA component

**Insight:** Cluster features dominate importance, proving that Part 1 segmentation was critical for Part 2 predictions.

### 3. Business Impact

**Current State (No Targeting):**
- Response rate: 1-2%
- Contact 100,000 people → 1,000-2,000 customers
- Cost: $100,000 (at $1 per contact)
- Cost per acquisition: $50-100

**With ML Targeting (Top 30%):**
- Predicted response rate: 4-6% (in high-value segments)
- Contact 30,000 people → 1,200-1,800 customers
- Cost: $30,000
- Cost per acquisition: $16-25
- **Savings: 70% reduction in cost per customer** 💰

### 4. Model Performance

**Validation Results:**
- **ROC-AUC:** 0.9709 (Excellent)
- **Precision@20%:** ~15% (vs. 1.5% baseline)
- **Lift@20%:** 10x improvement
- **Recall@20%:** 67% of all responders captured

**Comparison to Benchmarks:**
- Random selection: 0.50 ROC-AUC
- Simple heuristics: 0.55-0.60 ROC-AUC
- Basic logistic regression: 0.65-0.70 ROC-AUC
- **Our model: 0.97 ROC-AUC** ✅

## 🚀 Usage

### Running the Complete Pipeline

1. **Ensure data is in place:**
```bash
ls data/Udacity_*.csv  # Should show 4 CSV files
```

2. **Open Jupyter Notebook:**
```bash
jupyter notebook "Arvato Project Workbook.ipynb"
```

3. **Run cells sequentially:**
   - Part 0: Data Preprocessing (creates cleaned datasets)
   - Part 1: Customer Segmentation (creates visualizations)
   - Part 2: Response Prediction (trains models)
   - Part 3: Test Predictions (generates submission)

### Quick Start (Using Pre-processed Data)

If you've already run preprocessing once:
```python
import pandas as pd

# Load pre-processed data (much faster!)
azdias = pd.read_pickle('data/azdias_cleaned.pkl')
customers = pd.read_pickle('data/customers_cleaned.pkl')

# Load fitted models
import joblib
scaler = joblib.load('models/scaler.pkl')
pca = joblib.load('models/pca.pkl')
kmeans = joblib.load('models/kmeans.pkl')
model = joblib.load('models/best_model.pkl')

# Make predictions on new data
X_new_scaled = scaler.transform(X_new)
X_new_pca = pca.transform(X_new_scaled)
predictions = model.predict_proba(X_new_pca)[:, 1]
```

### Reproducing Results

All results are reproducible with fixed random seeds:
```python
random_state = 42  # Used throughout
```

**Expected Runtime:**
- Preprocessing: 15-20 minutes (first time only)
- Part 1 (Segmentation): 10-15 minutes
- Part 2 (Prediction): 20-30 minutes
- Total: ~1 hour for complete pipeline

## 📦 Dependencies

### Core Libraries
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
imbalanced-learn>=0.9.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
joblib>=1.1.0
```

### Optional (for enhanced functionality)
```
xgboost>=1.5.0         # Alternative classifier
shap>=0.40.0           # Model interpretability
plotly>=5.0.0          # Interactive visualizations
```

### Installing All Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy==1.21.6
pandas==1.3.5
scikit-learn==1.0.2
imbalanced-learn==0.9.1
matplotlib==3.5.3
seaborn==0.11.2
jupyter==1.0.0
joblib==1.1.1
openpyxl==3.0.10       # For reading Excel files
```

## 📈 Visualizations

The project generates several key visualizations:

1. **Missing Data Analysis** (`missing_data_analysis.png`)
   - Heatmap of missing value patterns
   - Helps inform feature selection strategy

2. **PCA Variance Analysis** (`pca_variance_analysis.png`)
   - Scree plot showing variance explained
   - Cumulative variance curve
   - Justifies component selection

3. **Optimal K Selection** (`kmeans_optimal_k.png`)
   - Elbow method curve
   - Silhouette scores
   - Supports K=14 choice

4. **Cluster Comparison** (`cluster_comparison_analysis.png`)
   - Side-by-side cluster distributions
   - Over/under-representation ratios
   - **Key insight visualization**

5. **Model Evaluation** (`model_evaluation.png`)
   - ROC curves
   - Precision-recall curves
   - Confusion matrices

6. **Feature Importance** (`feature_importance.png`)
   - Top 15 predictive features
   - Shows cluster features dominate

7. **Test Predictions** (`test_predictions_distribution.png`)
   - Distribution of predicted probabilities
   - Predicted class balance

## 🎓 Acknowledgments

### Data & Partnership
- **Bertelsmann Arvato Analytics** for providing real-world business data
- **Udacity** for project framework and support
- Dataset documentation: DIAS (Datenbank für Internationale Adressen und Soziodemografie)

### Methodology References
- Market Segmentation: Smith, W. R. (1956)
- Customer Analytics: Wedel & Kamakura (2000)
- Response Modeling: Neslin et al. (2006)
- Class Imbalance: Chawla et al. (2002) - SMOTE algorithm

### Technical Resources
- scikit-learn documentation
- imbalanced-learn documentation
- Kaggle community discussions

## 📄 License

This project is licensed under the MIT License - see below for details:
```
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Note:** The data is proprietary to Bertelsmann Arvato Analytics and subject to separate licensing terms.

## 📞 Contact

**Author:** [Your Name]  
**Email:** [shotadeaminat@gmail.com]  
**LinkedIn:** [https://www.linkedin.com/in/shotade-aminat/]  
**GitHub:** [github.com/mindelias]  
 

## 🔗 Related Projects

- [Customer Churn Prediction](link)
- [Market Basket Analysis](link)
- [Recommendation Systems](link)

## 📝 Citation

If you use this project or methodology in your research, please cite:
```bibtex
@misc{arvato_customer_segmentation_2024,
  author = {Aminat Shotade},
  title = {Customer Segmentation and Response Prediction for Arvato Financial Services},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/mindelias/bertelsmann_arvato_project}
}
```

---

**⭐ If you found this project helpful, please consider giving it a star!**

**Last Updated:** December 2024