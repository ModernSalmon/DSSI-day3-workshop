## DSSI Day 3 Workshop: Streamlit Community Cloud App

This project customizes the provided framework into a new machine learning application:

**Breast Cancer Screening Risk Checker**

It uses a locally trained `scikit-learn` model, persists the model and metadata, and serves predictions through a Streamlit web app that can be deployed on Streamlit Community Cloud.

### Project Structure
- `src/training.py`: trains and persists the classifier
- `src/model_registry.py`: versions and saves models and metadata
- `src/inference.py`: loads the latest trained model for prediction
- `src/data_processor.py`: prepares the dataset and inference inputs
- `app.py`: Streamlit user interface
- `data/breast_cancer_screening.csv`: local training dataset

### Requirements
1. Python 3.11+
2. GitHub account
3. Streamlit Community Cloud account
4. Virtual environment

### Step 1: Create and activate the environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Train and save the model
Run:
```bash
python -m src.training --data_path data/breast_cancer_screening.csv
```

This will:
- load the dataset
- scale the numeric features
- train a `RandomForestClassifier`
- evaluate it using recall and false discovery rate
- save the current best model into `models/`
- save metadata into `metadata/`

### Step 3: Run the Streamlit app locally
```bash
streamlit run app.py --server.port 8080
```

Then open:
```text
http://localhost:8080
```

### Step 4: Push to GitHub
If your remote is named `origin`:
```bash
git add .
git commit -m "Complete Streamlit workshop app"
git push -u origin main
```

If your remote is currently named `orgin`, either use that name or rename it first:
```bash
git remote rename orgin origin
git push -u origin main
```

### Step 5: Deploy on Streamlit Community Cloud
1. Go to [https://share.streamlit.io](https://share.streamlit.io)
2. Click `New app`
3. Select your GitHub repository
4. Set the main file path to `app.py`
5. Deploy

### Workshop Submission Checklist
Your slides should include:
- group members' names
- Streamlit application URL with sharing enabled
- screenshot of the running application

### Notes
- This is a workshop demo for educational use only.
- The output is a screening aid, not a real medical diagnosis.
