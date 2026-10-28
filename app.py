
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Page setup
st.set_page_config(page_title="StudySmart", page_icon="📚")

st.title("📚 StudySmart")
st.subheader("Student Performance Predictor")

st.write(
    "Predict an estimated exam score using study hours, "
    "attendance, and previous marks."
)

st.info("This beginner project uses simulated sample data.")

# Create sample student data
np.random.seed(42)
n = 300

hours = np.random.uniform(0, 8, n)
attendance = np.random.uniform(40, 100, n)
previous_marks = np.random.uniform(20, 100, n)
noise = np.random.normal(0, 6, n)

next_marks = (
    0.45 * previous_marks
    + 3.5 * hours
    + 0.25 * (attendance - 40)
    + noise
)

next_marks = np.clip(next_marks, 0, 100)

data = pd.DataFrame({
    "Study Hours": hours,
    "Attendance": attendance,
    "Previous Marks": previous_marks,
    "Next Exam Marks": next_marks
})

# Prepare features and target
X = data[["Study Hours", "Attendance", "Previous Marks"]]
y = data["Next Exam Marks"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the ML model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# User inputs
st.header("Enter Student Details")

study_hours = st.slider(
    "Daily study hours", 0.0, 16.0, 3.0, 0.5
)

attendance_input = st.slider(
    "Attendance (%)", 0, 100, 75
)

previous_input = st.slider(
    "Previous exam marks", 0, 100, 65
)

# Make prediction
if st.button("Predict My Score"):
    student = pd.DataFrame({
        "Study Hours": [study_hours],
        "Attendance": [attendance_input],
        "Previous Marks": [previous_input]
    })

    result = model.predict(student)[0]
    result = np.clip(result, 0, 100)

    st.success(f"Estimated exam score: {result:.1f} / 100")

    if result >= 75:
        st.write("Keep up your consistent study habits!")
    elif result >= 50:
        st.write("Regular revision may help improve your performance.")
    else:
        st.write("Consider a consistent study schedule and extra practice.")

# Model evaluation
st.divider()
st.header("Model Evaluation")

col1, col2 = st.columns(2)

col1.metric("Mean Absolute Error", f"{mae:.2f}")
col2.metric("R² Score", f"{r2:.3f}")

st.caption(
    "These metrics use simulated test data and do not establish "
    "accuracy on real student results."
)

# Preview sample data
with st.expander("View sample student data"):
   st.dataframe(data.head(10), width="stretch")
