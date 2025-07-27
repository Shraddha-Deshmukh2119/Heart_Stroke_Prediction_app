import streamlit as st
import pandas as pd
import joblib

st.markdown(
    """
    <style>
    body {
        background-color: #fff0f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Load model, scaler and expected columns
model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_cols = joblib.load('columns.pkl')

# UI
st.title('Heart Stroke Prediction ❤')
st.markdown('Provide the following details:')

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ['M', 'F'])
chest_pain = st.selectbox('Chest Pain Type', ['ATA', 'NAP', 'TA', 'ASY'])
resting_bp = st.number_input('Resting BP (mm/Hg)', 80, 200, 120)
cholesterol = st.number_input('Cholesterol (mg/dl)', 100, 600, 200)
fasting_bs = st.selectbox("FastingBS > 120 mg/dl", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
max_hr = st.slider("Max HR", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ['Y', 'N'])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'])

if st.button("Predict"):
    # Create the raw input dictionary with initial continuous values
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'OldPeak': oldpeak
    }

    # Add one-hot encoded fields
    one_hot_fields = {
        f'Sex_{sex}': 1,
        f'ChestPainType_{chest_pain}': 1,
        f'RestingECG_{resting_ecg}': 1,
        f'ExerciseAngina_{exercise_angina}': 1,
        f'ST_Slope_{st_slope}': 1
    }

    raw_input.update(one_hot_fields)

    # Create input DataFrame
    input_df = pd.DataFrame([raw_input])

    # Ensure all expected columns are present
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_cols]

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(scaled_input)[0]

    # Output
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
