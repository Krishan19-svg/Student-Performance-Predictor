# StudySmart: Student Performance Predictor

## About
StudySmart is a beginner-level Machine Learning web application that estimates exam scores using study hours, attendance, and previous exam marks.

## Features
- Interactive input sliders
- Exam score prediction
- Linear Regression model
- Model evaluation using MAE and R²
- Sample training data preview

## Technologies
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn

## How to Run
Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## Machine Learning Approach
The application generates 300 synthetic student records, splits them into training and testing sets, trains a Linear Regression model, and evaluates it on held-out test data.

## Limitations
The dataset is simulated using an assumed mathematical relationship. The model has not been validated on real student data, so the results should not be treated as reliable predictions of actual academic performance.

## Future Improvements
- Evaluate using a real student-performance dataset
- Compare multiple ML algorithms
- Add charts and data visualisations
- Improve the interface and model evaluation

## Purpose
Created as a first-year B.Tech CSE project for GDG college recruitment.
