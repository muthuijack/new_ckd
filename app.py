import streamlit as st
import pandas as pd
import mysql.connector
import hashlib
import os
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import io

# =====================================================
# 1. INITIALIZATION & ASSETS
# =====================================================
st.set_page_config(page_title="NephroAI Staging System", layout="wide", page_icon="🏥")

@st.cache_resource
def load_clinical_assets():
    model = tf.keras.models.load_model("models/ckd_staging_model.keras")
    scaler = joblib.load("models/staging_scaler.pkl")
    features = joblib.load("models/feature_columns.pkl")
    return model, scaler, features

try:
    staging_model, staging_scaler, feature_cols = load_clinical_assets()
except Exception as e:
    st.error("Model assets not found. Please run the training script first.")
    st.stop()

# =====================================================
# 2. DATABASE CORE (MYSQL)
# =====================================================
def get_db_connection():
    return mysql.connector.connect(
        host=st.secrets["mysql"]["host"],
        user=st.secrets["mysql"]["user"],
        password=st.secrets["mysql"]["password"],
        database=st.secrets["mysql"]["database"],
        port=st.secrets["mysql"]["port"]
    )

def init_db():
    db = get_db_connection()
    cursor = db.cursor()
    # Users Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(255) UNIQUE, 
            password TEXT, 
            role VARCHAR(50)
        )""")
    # Predictions Table (Now with ckd_stage for the graph)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions(
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(255), 
            ckd_stage INT,
            risk_score FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
    db.commit()
    db.close()

init_db()

# =====================================================
# 3. PATIENT PROGRESS GRAPH
# =====================================================
def plot_health_trend(username):
    db = get_db_connection()
    query = "SELECT created_at, ckd_stage FROM predictions WHERE username=%s ORDER BY created_at ASC"
    df_history = pd.read_sql(query, db, params=(username,))
    db.close()

    if not df_history.empty:
        st.subheader("📈 Your Clinical Progression Trend")
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Plotting the stages (0 to 5)
        ax.plot(df_history['created_at'], df_history['ckd_stage'], 
                marker='o', linestyle='-', color='#e63946', linewidth=2)
        
        ax.set_yticks([0, 1, 2, 3, 4, 5])
        ax.set_yticklabels(['Healthy', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4', 'Stage 5'])
        ax.set_title(f"Kidney Health History for {username}")
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45)
        
        st.pyplot(fig)
    else:
        st.info("No historical records found yet. Run your first analysis to see the trend!")

# =====================================================
# 4. DIAGNOSTIC INTERFACE
# =====================================================
def patient_page():
    st.title(f"👋 Welcome to your Health Portal, {st.session_state.username}")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        with st.form("diagnostic_form"):
            st.subheader("🧪 Input Clinical Markers")
            inputs = {}
            # Dynamically create inputs based on features the model was trained on
            c1, c2 = st.columns(2)
            for i, col_name in enumerate(feature_cols):
                with (c1 if i % 2 == 0 else c2):
                    inputs[col_name] = st.number_input(f"{col_name.replace('_', ' ').title()}", value=0.0)
            
            submit = st.form_submit_button("Run Deep Learning Analysis")

        if submit:
            # Prepare data for model
            input_df = pd.DataFrame([inputs])[feature_cols]
            scaled_data = staging_scaler.transform(input_df)
            
            # Predict Stage (0-5)
            probs = staging_model.predict(scaled_data)
            pred_stage = int(np.argmax(probs[0]))
            risk_val = float(np.max(probs[0])) # Confidence score
            
            # Save to Database
            db = get_db_connection()
            cursor = db.cursor()
            cursor.execute(
                "INSERT INTO predictions (username, ckd_stage, risk_score) VALUES (%s, %s, %s)",
                (st.session_state.username, pred_stage, risk_val)
            )
            db.commit()
            db.close()
            
            st.success(f"Analysis Complete! Predicted Status: Stage {pred_stage}" if pred_stage > 0 else "Analysis Complete! Result: Healthy")
            st.rerun()

    with col2:
        # Show the progression graph
        plot_health_trend(st.session_state.username)
        
        # Display latest result details
        st.markdown("""
        ### Understanding Your Results
        - **Healthy (0):** Normal kidney function.
        - **Stage 1-2:** Mild decrease in function.
        - **Stage 3:** Moderate damage; medical intervention required.
        - **Stage 4-5:** Severe damage; specialist care needed.
        """)

# =====================================================
# 5. AUTHENTICATION & ROUTING
# =====================================================
def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        # Simplified Login for Demo
        st.title("🏥 NephroAI Login")
        user = st.text_input("Username")
        if st.button("Login / Register"):
            # Auto-register/login logic for simplicity
            db = get_db_connection()
            cursor = db.cursor()
            cursor.execute("SELECT * FROM users WHERE username=%s", (user,))
            if not cursor.fetchone():
                cursor.execute("INSERT INTO users (username, role) VALUES (%s, %s)", (user, 'patient'))
                db.commit()
            db.close()
            st.session_state.logged_in = True
            st.session_state.username = user
            st.rerun()
    else:
        st.sidebar.button("Logout", on_click=lambda: st.session_state.update({"logged_in": False}))
        patient_page()

if __name__ == "__main__":
    main()