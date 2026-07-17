# Rare Event Multi-Scale Prediction

A machine learning framework for rare-event binary classification using approximately ten years of longitudinal National Football League (NFL) data to investigate whether macro-level field conditions can predict player injuries.

This project analyzes three independent NFL datasets spanning more than 280,000 plays and evaluates whether environmental, tactical, and game-level variables contain sufficient predictive signal to forecast injury occurrence. Because injury events are exceptionally rare (ranging from approximately 0.02% to 2.44% depending on the dataset), the project focuses on the unique challenges of highly imbalanced classification problems.

The repository documents the complete data science workflow—from exploratory data analysis through model development and evaluation—and includes both a technical report and a business report communicating the findings to different audiences.

---

## Repository Structure

```
Rare-Event-Multi-Scale-Prediction/
│
├── README.md
│
├── Reports/
│   ├── Technical_Report.md
│   ├── Technical_Report.pdf
│   ├── Business_Report.md
│   └── Business_Report.pdf
│
├── Dataset Analysis/
│   ├── Big Data Bowl EDA.ipynb
│   ├── NFL First and Future EDA.ipynb
│   └── Punt Data Analytics EDA.ipynb
│
├── Data Cleaning/
│   ├── Big Data Bowl Cleaning.ipynb
│   ├── NFL First and Future Cleaning.ipynb
│   └── Punt Data Analytics Cleaning.ipynb
│
├── Modeling/
│   ├── Feature Engineering.ipynb
│   ├── Feature Selection.ipynb
│   ├── Decision Trees.ipynb
│   ├── Random Forest.ipynb
│   ├── XGBoost.ipynb
│   ├── CatBoost.ipynb
│   └── Model Evaluation.ipynb
│
└── Images/
```

---

## Reports

Two versions of the final report are included.

### Technical Report

Written for data scientists and machine learning practitioners, this report provides a detailed discussion of the methodology used throughout the project. It includes:

- Problem formulation
- Data quality assessment
- Exploratory Data Analysis (EDA)
- Feature engineering
- Dimensionality reduction
- Model selection
- Hyperparameter optimization
- Cross-validation strategy
- Model evaluation
- Discussion and future research directions

Both Markdown and PDF versions are provided.

### Business Report

The Business Report communicates the same project and findings using language intended for business leaders and non-technical audiences. Rather than focusing on implementation details, it emphasizes:

- Business motivation
- Project objectives
- High-level methodology
- Major findings
- Actionable recommendations
- Future opportunities

Again, both Markdown and PDF versions are available.

---

## Datasets

This project combines three publicly available NFL datasets.

| Dataset | Purpose |
|----------|---------|
| **NFL First and Future** | Lower-extremity injury analysis, playing surfaces, environmental factors |
| **NFL Big Data Bowl** | Play-level football analytics and player tracking |
| **NFL Punt Data Analytics** | Punt-specific injury events and contextual game information |

Collectively, these datasets span approximately ten years of NFL play and include hundreds of thousands of observations.

---

## Jupyter Notebook Corpus

The repository is organized to mirror the complete machine learning workflow.

### Exploratory Data Analysis

Each dataset is independently explored to understand its structure, data quality, feature distributions, and potential relationships to injury occurrence.

- Big Data Bowl EDA
- NFL First and Future EDA
- Punt Data Analytics EDA

### Data Cleaning & Preprocessing

Data preparation notebooks document the complete preprocessing pipeline, including:

- Missing value treatment
- Standardization of categorical variables
- Feature encoding
- Outlier handling
- Multicollinearity analysis
- Feature selection
- Preparation for model training

### Modeling & Evaluation

Model development notebooks progress from baseline methods through modern ensemble approaches.

Models evaluated include:

- Decision Trees
- Random Forests
- Bagging Classifiers
- XGBoost
- CatBoost

Each notebook documents the modeling decisions, evaluation metrics, hyperparameter tuning process, interpretation of results, and opportunities for future improvement.

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- CatBoost
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Key Findings

While several environmental and tactical variables demonstrated statistically significant associations with injury occurrence, no combination of macro-level field conditions produced sufficient predictive power to reliably forecast injuries on their own.

The findings suggest that player biomechanics at the moment of impact likely contain substantially more predictive signal than contextual game conditions alone, providing a clear direction for future research.

---

## About This Repository

This repository was developed as the capstone project for Boston University's Master of Science in Data Science program.

Its purpose is to demonstrate an end-to-end machine learning workflow, emphasizing reproducible analysis, statistical reasoning, model development, technical communication, and translation of technical findings into business recommendations.