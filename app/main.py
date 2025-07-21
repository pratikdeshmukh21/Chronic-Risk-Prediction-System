# app/main.py
import streamlit as st
import pickle
import pandas as pd, numpy as np
from pathlib import Path
import plotly.graph_objects as go

MODEL_DIR = Path(__file__).parent.parent / 'models'
DISEASES = {
    'Diabetes':            ('diabetes_logistic_regression_model.pkl',  'Outcome'),
    'Heart Disease':       ('heart_gb.pkl',     'target'),
    'Chronic Kidney':      ('ckd_gb.pkl',       'class'),
    'Liver Disease':       ('liver_gb.pkl',     'Selector'),
    'Stroke':              ('stroke_gb.pkl',    'stroke'),
    'Hypertension':        ('htn_gb.pkl',       'Hypertension')
}

@st.cache_resource
def load_model(file):
    try:
        with open(MODEL_DIR / file, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Model file '{file}' not found in 'models/'. Please upload the required model.")
        return None
    except Exception as e:
        st.error(f"Error loading model '{file}': {e}")
        return None

def user_form(required_fields):
    cols = st.columns(2)
    user_data = {}
    for i, (f, dtype) in enumerate(required_fields.items()):
        with cols[i % 2]:
            if dtype == 'num':
                user_data[f] = st.number_input(f, step=0.1)
            else:
                user_data[f] = st.selectbox(f, dtype)
    return pd.DataFrame([user_data])

def handle_diabetes():
    diabetes_fields = {
        'age': ['20-30','30-40','40-50','50-60','60-70','70-80','80-90'],
        'sex': ['Male', 'Female'],
        'family_history': ['Yes', 'No'],
        'blood_pressure_status': ['Normal', 'Prehypertension', 'Hypertension'],
        'history_high_blood_sugar': ['Yes', 'No'],
        'physical_activity': ['Low', 'Moderate', 'High'],
        'smoking_status': ['Never', 'Former', 'Current']
    }
    df_user = user_form(diabetes_fields)
    results = {}
    if st.button('Predict'):
        model, _ = DISEASES['Diabetes']
        clf = load_model(model)
        if clf is None:
            results['Diabetes'] = 'Model not found'
        else:
            try:
                df_user['BMI'] = 25  # Set default BMI
                proba = clf.predict_proba(df_user)[0,1]
                results['Diabetes'] = round(proba*100, 1)
            except Exception as e:
                st.error(f"Prediction failed for Diabetes: {e}")
                results['Diabetes'] = 'Prediction error'
    return results

st.title("Chronic Disease Risk Dashboard")
chosen = st.multiselect("Select diseases to screen", list(DISEASES.keys()))
if chosen:
    results = {}
    if chosen == ['Diabetes']:
        results = handle_diabetes()
    else:
        # If Diabetes and others are selected, merge fields (or handle separately)
        merged_fields = {}
        if 'Diabetes' in chosen:
            merged_fields.update({
                'age': ['20-30','30-40','40-50','50-60','60-70','70-80','80-90'],
                'sex': ['Male', 'Female'],
                'family_history': ['Yes', 'No'],
                'blood_pressure_status': ['Normal', 'Prehypertension', 'Hypertension'],
                'history_high_blood_sugar': ['Yes', 'No'],
                'physical_activity': ['Low', 'Moderate', 'High'],
                'smoking_status': ['Never', 'Former', 'Current']
            })
        df_user = user_form(merged_fields)
        if st.button('Predict'):
            for d in chosen:
                model, _ = DISEASES[d]
                clf = load_model(model)
                if clf is None:
                    results[d] = 'Model not found'
                    continue
                try:
                    if d == 'Diabetes' and 'BMI' not in df_user.columns:
                        df_user['BMI'] = 25  # Set default BMI for Diabetes
                    proba = clf.predict_proba(df_user)[0,1]
                    results[d] = round(proba*100, 1)
                except Exception as e:
                    st.error(f"Prediction failed for {d}: {e}")
                    results[d] = 'Prediction error'
    st.subheader("Risk Summary (%)")
    if results:
        st.table(pd.DataFrame([
            {'Disease': k, 'Risk (%)': v} for k, v in results.items()
        ]))
