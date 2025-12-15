# Student Career Prediction System

This project predicts suitable career paths for students in the Computer Science domain using machine learning techniques.

We have developed an interactive **Web Application** using Streamlit, allowing users to assess themselves and receive instant, AI-driven career recommendations.

## Project Structure

```text
STUDENT_CAREER/
│
├── data/
│   ├── cleaned/            # Cleaned dataset
│   ├── mapping/            # Dataset used for career mapping
│   └── raw/                # Original raw data
│
├── notebooks/
│   ├── CareerMapping/           # Experiments on the initial dataset
│   ├── CareerMapping1/          # Experiments on the revised dataset
│
├── reports/
│   ├── BI.pbix             # Power BI report
│   
├── src/                    # Source code (training, preprocessing, utils)
│   ├── app.py                   # Main Streamlit application file
│   ├── career_prediction_model.pkl  # Trained Random Forest Model (from cleaned data)
│   ├── label_encoder.pkl        # Encoder for decoding predictions
│   ├── scaler.pkl               # Scaler for normalizing user input
│   └── snaptik.vn_Zrnji.mp4     # Video asset for the Custom Error Page
│
├── venvda/                 # Virtual environment (ignored by Git)
├── .gitignore
├── README.md
└── requirements.txt
```

## Data Analysis & Selection Strategy

To build a reliable system, we analyzed and evaluated datasets sourced from the open-source repository **[Career Prediction Using Machine Learning](https://github.com/TuhinPatra633/Career-Prediction-Using-Machine-Learning)** by **TuhinPatra633**.

### 1. Feature Engineering & Preprocessing**
We did not use the raw data directly. Instead, we implemented a robust preprocessing pipeline to clean the data and engineer new features (calculating aggregated `tech_score` and `soft_skill_score`).

We applied this identical pipeline to both raw datasets to generate two **processed candidates** for benchmarking:
* **`processed_raw_CareerMapping_with_scores.csv`**
- *Source:* Derived from the initial `CareerMapping.csv`.
- *Status:* Prepared for experimental comparison.
* **`processed_raw_CareerMapping1_with_scores.csv`**
- *Source:* Derived from the revised `CareerMapping1.csv`.
- *Status:* Prepared for experimental comparison.

> **Note:** These processed files are stored in the `data/cleaned/` directory and include the normalized scores required for our specific model architecture.

### 2. Comparative Experiment & Selection**
We trained models on **both** processed datasets to evaluate their realism and reliability:

* **Experiment A (using `processed_raw_CareerMapping_with_scores.csv`):**
    * *Result:* Models achieved ~100% accuracy.
    * *Verdict:* **Rejected**. The perfect score indicates overfitting/synthetic patterns, making it unsuitable for real-world predictions.
* **Experiment B (using `processed_raw_CareerMapping1_with_scores.csv`):**
    * *Result:* Models achieved realistic accuracy (~89%).
    * *Verdict:* **Selected**. This dataset reflects the complexity of real user profiles, ensuring the app provides genuine advice rather than memorized answers.


## Model Performance

After preprocessing, we trained and evaluated multiple machine learning models on our selected dataset (**`processed_raw_CareerMapping1_with_scores.csv`**). The table below summarizes the **Test Accuracy**, which measures how well each model generalizes to new, unseen data.

| Algorithm | Test Accuracy |
| :--- | :--- |
| Decision Tree | 79.1% |
| **Random Forest** | **78.2%** |
| XGBoost | 77.4% |
| SVM | 73.6% |
| KNN | 31.6% |
| Naïve Bayes | 19.0% |

### Model Selection: Why Random Forest?

Although **Decision Tree** models may occasionally show higher accuracy on specific test splits, we deliberately selected **Random Forest** for the final deployment. This decision is based on two critical factors for a real-world application:

1.  **Generalization vs. Overfitting:**
    * **Decision Trees** tend to create overly complex structures that "memorize" the noise in the training data (High Variance). While this yields high training scores, it often leads to poor performance on new, unseen user data.
    * **Random Forest** is an **Ensemble method** (Bagging). By aggregating predictions from multiple trees, it cancels out individual errors and focuses on the underlying patterns, ensuring the model works well for diverse student profiles.

2.  **Stability & Robustness:**
    * In a production Web App, user inputs can be unpredictable. Random Forest is significantly **more stable** than a single Decision Tree. Small changes in input data do not drastically flip the prediction, making the system more reliable for users.

> **Verdict:** We prioritize **Reliability** and **Real-world Performance** over raw Accuracy on paper. Random Forest offers the best balance for a robust Career Prediction System.


## Deployment Decision
Based on the comparative analysis above, the **Random Forest Classifier** demonstrated the highest stability and accuracy (~89%).

* **Final Training:** We re-trained the Random Forest model on the full `processed_raw_CareerMapping1_with_scores.csv` dataset to maximize its learning.
* **Export:** The trained model was serialized and saved as **`career_prediction_model.pkl`**.
* **Deployment:** This specific model file is now integrated into our **Streamlit Web Application** (`app.py`) to provide real-time career recommendations.


## How to Run the App Locally

### 1. Prerequisites
Ensure your system meets the following requirements:
* **Python 3.8** or higher installed.
* **Git** installed.

### 2. Installation
Open your terminal or command prompt and run the following commands to clone the repository and install dependencies:

```bash
# Step 1: Clone the repository
git clone [https://github.com/dilambestall/student-career-prediction-system.git](https://github.com/dilambestall/student-career-prediction-system.git)

# Step 2: Navigate to the project directory
cd student-career-prediction-system

# Step 3: Install required Python packages
pip install -r requirements.txt
```

### 3. Verify System Files
Before running the application, navigate to the src/ directory and ensure the following essential files are present. These files are required for the AI model and the Error Page to function correctly.
* app.py (Main Application)
* career_prediction_model.pkl (Trained Random Forest Model)
* label_encoder.pkl (Label Decoder)
* scaler.pkl (Input Normalizer)
* snaptik.vn_Zrnji.mp4 (Asset for the Custom Error Page)

### 4. Launch the Application
To start the web application, run the following commands in your terminal:
```bash
# 1. Navigate to the source code folder
cd src

# 2. Run the Streamlit app
streamlit run app.py
```

After executing the command, the application will automatically launch in your default web browser at: http://localhost:8501


