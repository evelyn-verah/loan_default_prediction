# 💳 FinTech Loan Default Prediction System
- AI-powered credit risk modeling pipeline** that predicts loan defaults using borrower and loan characteristics. Built with the CRISP-DM framework, this project demonstrates how machine learning can support responsible lending decisions, reduce default risk, and improve financial system resilience.
---

## 📑 Table of Contents
- [1. Business Understanding](#1-business-understanding)
- [2. Data Understanding](#2-data-understanding)
- [3. Data Preparation](#3-data-preparation)
- [4. Modeling](#4-modeling)
- [5. Evaluation](#5-evaluation)
- [6. Deployment Potential](#6-deployment-potential)
- [7. Real-World Application](#7-real-world-application)
- [8. Industry Relevance](#8-industry-relevance)
- [9. Key Innovations](#9-key-innovations)
- [Repository Structure](#repository-structure)
- [Technologies](#technologies)
- [Acknowledgements](#acknowledgements)

---

## 1. Problem statement
The objective was to build a **loan default prediction system** that enables financial institutions to:
- Identify **high-risk borrowers** before approving loans.
- Reduce **non-performing loans (NPLs)** and associated losses.
- Ensure **fair, transparent, and explainable lending practices**.

Key question: *Can borrower and loan features predict whether a loan will default?*

---

## 2. Data Understanding
- **Source**: Lending Club financial dataset.
- **Size**: 142 features across thousands of loan records.
- <img width="526" height="451" alt="image" src="https://github.com/user-attachments/assets/1375d560-5476-46da-b7d0-52a559b55689" />
- **Target Variable Remapping**
- <img width="2042" height="712" alt="image" src="https://github.com/user-attachments/assets/a50de6f0-339c-4df3-9c3c-7e937edd604f" />
- **Binary Classification of target variable**
- <img width="296" height="226" alt="image" src="https://github.com/user-attachments/assets/d5ebb4d7-8e7a-4f08-944a-84391ae50e41" />
- **Summary Statistics**
- <img width="2048" height="488" alt="image" src="https://github.com/user-attachments/assets/426da16e-0a82-4dba-a5fe-a600fc327d2d" />
- **Distribution of features**
- <img width="2048" height="1507" alt="image" src="https://github.com/user-attachments/assets/c1a25d0d-9535-4da2-8fd4-9308f818e583" />

### 3. Exploratory Data Analysis:
- <img width="1716" height="910" alt="image" src="https://github.com/user-attachments/assets/27735f75-9c8e-4aa4-92d9-d149c8acee72" />
- <img width="1736" height="824" alt="image" src="https://github.com/user-attachments/assets/ee300c57-2572-4ea3-a241-3dab536cfa43" />
- <img width="1872" height="920" alt="image" src="https://github.com/user-attachments/assets/d3c7025c-e7f3-4d3c-8491-ad6b55e4d296" />
- <img width="1858" height="954" alt="image" src="https://github.com/user-attachments/assets/fb6ca7be-f68e-40c6-9e88-a6b8e4f9c4e7" />
- <img width="1856" height="1012" alt="image" src="https://github.com/user-attachments/assets/7cf5ca10-658b-4359-ab1f-fc4d98891d29" />
- <img width="2048" height="1574" alt="image" src="https://github.com/user-attachments/assets/03489afd-2417-4658-a8cf-264bd2918059" />

## 3. Data Preparation
- **Feature Reduction**: Dropped IDs, features with >50% missing values, and leakage variables (e.g., hardship flags).
- **Missing Values**: Imputed using mean (numeric) and mode (categorical).
- **Encoding**:
  - One-hot encoding for nominal categorical variables (`purpose`).
  - Label encoding for ordinal categorical variables (`grade`).
- **Resampling**: Used **RandomUnderSampler** to address class imbalance.
- **Train/Test Split**: 80/20 with stratification to preserve class distribution.

---

## 4. Modeling
Implemented three supervised machine learning models:
- **Logistic Regression** – baseline, interpretable, robust.
- <img width="1624" height="276" alt="image" src="https://github.com/user-attachments/assets/13ccbdf3-2339-4bcd-976e-bb069de98b4a" />

- **Decision Tree** – captures non-linearities, interpretable structure.
- <img width="1794" height="254" alt="image" src="https://github.com/user-attachments/assets/d8263162-c896-4a05-a494-caf3644cdc4a" />

- **K-Nearest Neighbors (KNN)** – tested as a non-parametric approach.
- <img width="1456" height="266" alt="image" src="https://github.com/user-attachments/assets/00718e0e-be00-4d52-abeb-71f2335a266f" />

Each model was tuned for hyperparameters and evaluated using ROC-AUC, accuracy, precision, and recall.

---

## 5. Evaluation
- **Best model**: Logistic Regression, due to **strong recall** (catching defaults) and interpretability for stakeholders.
- **Decision Trees** were useful for visualizing decision rules.
- **KNN** underperformed on large, sparse data.

📊 **Results Table**:

| Model               | Accuracy | Precision | Recall | ROC-AUC |
|---------------------|----------|-----------|--------|---------|
| Logistic Regression | 0.87     | 0.83      | 0.85   | 0.85    |
| Decision Tree       | 0.82     | 0.79      | 0.81   | 0.78    |
| KNN                 | 0.78     | 0.75      | 0.72   | 0.74    |

- ROC-AUC Output
- <img width="1190" height="898" alt="image" src="https://github.com/user-attachments/assets/aae4c9fc-47d6-4fda-a2c5-9bdacf368ef8" />

## 📊 Business Case & project impact

- **Default Reduction** → Predict fewer defaults, saving financial losses.  
- **Revenue Growth** → Increase approved loans for low-risk borrowers, boosting loan volume.  
- **Cost Savings** → Reduce manual assessments and lower bad debt.  
- **ROI** → Compare cost savings and additional revenue to model development and deployment costs.  
---

## 6. Deployment Potential
- **Automated Loan Approval**: Integrate the model into loan origination systems for real-time borrower scoring.
- **Portfolio Risk Management**: Assess aggregate risk exposure and stress-test scenarios.
- **Dynamic Loan Pricing**: Adjust interest rates or terms based on predicted default risk.
- **Monitoring Dashboard**: Provide risk officers with alerts on default probability shifts.

---

## 7. Real-World Application
- **Credit Risk Assessment** → Banks and credit unions can identify high-risk borrowers before loan approval.
- **Portfolio Management** → Lenders can monitor risk exposure at the portfolio level and reduce non-performing loans.
- **Dynamic Loan Pricing** → Institutions can tailor interest rates and loan terms based on predicted risk.
- **Regulatory Compliance** → Models provide explainable decisions that meet fair lending standards.
---

## 8. Industry Relevance
- **FinTech Growth** → As digital lending expands in the U.S., automated credit risk tools are essential.
- **Responsible AI in Finance** → Emphasis on explainable models supports fair access to credit and avoids bias.
- **Systemic Stability** → Reducing loan defaults strengthens the financial system’s resilience.
- **Policy Alignment** → Aligns with regulatory priorities like transparency, consumer protection, and risk management.

---

## 9. Key Innovations
- **CRISP-DM Framework** → Structured, industry-standard process for building trustable models.
- **Balanced Class Handling** → Techniques like undersampling improve performance on rare but costly defaults.
- **Interpretability** → Logistic Regression and Decision Trees provide human-readable risk scores, unlike black-box models.
- **Deployment Focus** → Designed with potential integration into loan origination and credit monitoring systems.

---

## 📂 Repository Structure
```text
loan-default-prediction/
├─ data/                        # Sample or synthetic Lending Club data
├─ notebooks/                   # CRISP-DM workflow
│  ├─ 01_data_understanding.ipynb
│  ├─ 02_data_preparation.ipynb
│  ├─ 03_modeling.ipynb
│  ├─ 04_evaluation.ipynb
├─ docs/                        # Reports and slides
│  ├─ Team5_Lending Club Financial Data.pdf
│  ├─ Loan default prediction report.pdf
│  └─ Deck.pdf
├─ requirements.txt
├─ README.md
└─ LICENSE
---
```

## 🛠 Technologies
- Python, Pandas, NumPy
- scikit-learn (ML models)
- imbalanced-learn (sampling strategies)
- Matplotlib & Seaborn (visualizations)
- Jupyter Notebook

---

## 🙏 Acknowledgements
- **Lending Club dataset** for financial data.
- **Open-source tools**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [scikit-learn](https://scikit-learn.org/), [imbalanced-learn](https://imbalanced-learn.org/), [Matplotlib](https://matplotlib.org/).
- **Faculty and mentors** for guidance on credit risk modeling and ethical AI practices.
