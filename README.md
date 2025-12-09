# Student Career Prediction System

This project predicts suitable career paths for students in the Computer Science domain using machine learning techniques.

## 🔍 Overview
- Data collected from LinkedIn and Google Forms.
- Preprocessed to remove invalid or duplicate entries.
- Trained and evaluated multiple ML models for accuracy.
- Best-performing algorithm selected for deployment.

## 🧠 Technologies Used
- Python (pandas, scikit-learn, matplotlib)
- Jupyter Notebook
- Git & GitHub

## 📁 Project Structure

```text
STUDENT_CAREER/
│
├── data/
│   ├── cleaned/            # Cleaned dataset
│   ├── mapping/            # Dataset used for career mapping
│   └── raw/                # Original raw data
│
├── notebooks/
│   ├── Mapping.ipynb       # Main notebook for career mapping
│   └── Mapping1.ipynb      # Additional mapping notebook
│
├── reports/
│   ├── BI.pbix             # Power BI report
│   
├── src/                    # Source code (training, preprocessing, utils)
│   └── ...
│
├── venvda/                 # Virtual environment (ignored by Git)
├── .gitignore
├── README.md
└── requirements.txt
```


## 🚀 How to Run

```bash
# Clone this repository
git clone https://github.com/<your-username>/student-career-prediction-system.git

# Install dependencies
pip install -r requirements.txt
