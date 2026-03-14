# Titanic Survival Predictor 🚢

A machine learning project that predicts whether a passenger survived the Titanic disaster based on features like age, gender, and passenger class.

---

## Problem Statement

On April 15, 1912, the Titanic sank after colliding with an iceberg. Out of 2,224 passengers, only 710 survived. This project builds a machine learning model that learns patterns from passenger data and predicts survival with **83% accuracy**.

---

## Demo

```
predict_survival(pclass=1, sex='female', age=28, fare=100)
# Output: SURVIVED (survival probability: 89%)

predict_survival(pclass=3, sex='male', age=22, fare=7)
# Output: DIED (survival probability: 14%)
```

---

## Key Findings

| Feature | Insight |
|---|---|
| Gender | Women had 74% survival rate vs 19% for men |
| Passenger Class | 1st class had 63% survival rate vs 24% for 3rd class |
| Age | Children had higher survival priority |
| Fare | Higher fare = higher survival chance |

---

## Model Results

| Model | Test Accuracy | CV Accuracy |
|---|---|---|
| Logistic Regression | 79.9% | 79.3% |
| Random Forest | **83.2%** | **81.9%** |

> Best model: **Random Forest Classifier**

---

## Tech Stack

- **Language:** Python 3
- **Data:** Titanic dataset (seaborn built-in)
- **Libraries:**
  - `pandas` — data manipulation
  - `numpy` — numerical operations
  - `scikit-learn` — ML models and evaluation
  - `matplotlib` + `seaborn` — data visualization

---

## Project Structure

```
titanic-survival-predictor/
│
├── titanic_predictor.py    # Main ML pipeline
├── README.md               # Project documentation
```

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/priyanshuyad1207-rgb/titanic-survival-predictor.git
cd titanic-survival-predictor
```

**2. Install dependencies**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

**3. Run the project**
```bash
python titanic_predictor.py
```

---

## What This Project Covers

- Exploratory Data Analysis (EDA)
- Data cleaning and missing value handling
- Feature engineering (family size, is alone)
- Model training and comparison
- Cross validation
- Confusion matrix and feature importance
- Custom prediction function

---

## Output Charts

The project generates 3 charts automatically:

- `exploration.png` — survival rate by gender, class, and age
- `confusion_matrix.png` — model prediction accuracy breakdown
- `feature_importance.png` — which features mattered most

---

## What I Learned

- How to build a complete ML pipeline from raw data to predictions
- Feature engineering can significantly improve model accuracy
- Random Forest outperforms Logistic Regression on this dataset
- Gender and fare were the strongest predictors of survival

---

## Future Improvements

- [ ] Extract title (Mr, Mrs, Miss) from passenger names as a feature
- [ ] Try XGBoost for better accuracy
- [ ] Build a web app using Streamlit so anyone can predict survival
- [ ] Hyperparameter tuning with GridSearchCV

---

## Author

**Priyansh** — Aspiring ML Engineer  
Learning machine learning to build real-world AI projects.

[![GitHub](https://img.shields.io/badge/GitHub-priyanshuyad1207--rgb-blue)](https://github.com/priyanshuyad1207-rgb)

---

*This is my first ML project — part of my journey to becoming an ML Engineer* 🚀

