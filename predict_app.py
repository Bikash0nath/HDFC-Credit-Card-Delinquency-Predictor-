# predict_app.py
# Run: streamlit run predict_app.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="Credit Card Delinquency Predictor",
    page_icon="💳",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model_and_scaler(model_path='credit_delinq_model.pkl', scaler_path='scaler.pkl'):
    """Load the trained model and scaler"""
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found!")
        st.stop()
    if not os.path.exists(scaler_path):
        st.error(f"❌ Scaler file '{scaler_path}' not found!")
        st.stop()
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def preprocess_data(data):
    """Rename columns and prepare features"""
    # Strip all column names to remove \n and extra spaces
    data.columns = data.columns.str.strip()
    
    # Rename columns to match model training
    column_mapping = {
        'Customer_ID': 'Customer_ID',
        'Credit_Limit': 'Credit_Limit',
        'Utilisation_%': 'Utilisation_%',
        'Avg_Payment_Ratio': 'Avg_Payment_Ratio',
        'Min_Due_Paid_Frequency': 'Min_Due_Paid_Frequency',
        'Merchant_Mix_Index': 'Merchant_Mix_Index',
        'Cash_Withdrawal_%': 'Cash_Withdrawal_%',
        'Recent_Spend_Change_%': 'Recent_Spend_Change_%',
        'Delinquency_Flag_Next_Month(DPD_Bucket)': 'DPD_Bucket_Next_Month'
    }
    
    # Apply renaming
    data.rename(columns=column_mapping, inplace=True)
    
    # Remove target column if present
    if 'DPD_Bucket_Next_Month' in data.columns:
        data = data.drop('DPD_Bucket_Next_Month', axis=1)
    
    # Save customer IDs
    customer_ids = data['Customer_ID'].copy()
    data = data.drop('Customer_ID', axis=1)
    
    # Select only the 7 features in correct order
    required_features = [
        'Credit_Limit', 
        'Utilisation_%', 
        'Avg_Payment_Ratio',
        'Min_Due_Paid_Frequency', 
        'Merchant_Mix_Index',
        'Cash_Withdrawal_%', 
        'Recent_Spend_Change_%'
    ]
    
    data_features = data[required_features]
    
    return data_features, customer_ids


def make_predictions(model, scaler, features, customer_ids):
    """Scale data and make predictions"""
    # Convert DataFrame to numpy array
    features_array = features.to_numpy()
    
    # Scale the data
    features_scaled = scaler.transform(features_array)
    
    # Make predictions
    predictions = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Customer_ID': customer_ids.values,
        'Predicted_DPD_Bucket': predictions,
    })
    
    # Add risk level
    results['Risk_Level'] = results['Predicted_DPD_Bucket'].map({
        0: 'No Risk',
        1: 'Low Risk (1-30 DPD)',
        2: 'Medium Risk (31-60 DPD)',
        3: 'High Risk (61+ DPD)'
    })
    
    # Add probabilities for each bucket
    results['Prob_No_Risk'] = probabilities[:, 0].round(4)
    results['Prob_1-30_DPD'] = probabilities[:, 1].round(4)
    results['Prob_31-60_DPD'] = probabilities[:, 2].round(4)
    results['Prob_61+_DPD'] = probabilities[:, 3].round(4)
    
    return results


def main():
    # Header
    st.markdown('<h1 class="main-header">💳 Credit Card Delinquency Predictor</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 📊 Upload customer data to predict delinquency risk
    Upload an Excel or CSV file containing customer credit card information to get instant risk predictions.
    """)
    
    # Load model and scaler
    with st.spinner("Loading model..."):
        model, scaler = load_model_and_scaler()
    st.success("✅ Model loaded successfully!")
    
    # File upload section
    st.markdown("---")
    st.subheader("📁 Upload Customer Data")
    
    uploaded_file = st.file_uploader(
        "Choose a file (Excel or CSV)",
        type=['xlsx', 'xls', 'csv'],
        help="File should contain columns: Customer_ID, Credit_Limit, Utilisation_%, Avg_Payment_Ratio, Min_Due_Paid_Frequency, Merchant_Mix_Index, Cash_Withdrawal_%, Recent_Spend_Change_%"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                try:
                    data = pd.read_excel(uploaded_file, sheet_name='Sample')
                    st.info("✅ Loaded 'Sample' sheet")
                except:
                    data = pd.read_excel(uploaded_file)
                    st.info("✅ Loaded first sheet")
            
            st.success(f"✅ File loaded successfully! Shape: {data.shape}")
            
            # Show preview
            with st.expander("📋 Preview uploaded data (first 10 rows)"):
                st.dataframe(data.head(10))
            
            # Preprocess data
            with st.spinner("Preprocessing data..."):
                features, customer_ids = preprocess_data(data)
            
            st.success(f"✅ Preprocessed {len(features)} customers")
            
            # Make predictions
            with st.spinner("Making predictions..."):
                results = make_predictions(model, scaler, features, customer_ids)
            
            st.success("✅ Predictions complete!")
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            # Risk summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            risk_counts = results['Risk_Level'].value_counts()
            total = len(results)
            
            with col1:
                no_risk = risk_counts.get('No Risk', 0)
                st.metric("🟢 No Risk", f"{no_risk}", f"{no_risk/total*100:.1f}%")
            
            with col2:
                low_risk = risk_counts.get('Low Risk (1-30 DPD)', 0)
                st.metric("🟡 Low Risk", f"{low_risk}", f"{low_risk/total*100:.1f}%")
            
            with col3:
                med_risk = risk_counts.get('Medium Risk (31-60 DPD)', 0)
                st.metric("🟠 Medium Risk", f"{med_risk}", f"{med_risk/total*100:.1f}%")
            
            with col4:
                high_risk = risk_counts.get('High Risk (61+ DPD)', 0)
                st.metric("🔴 High Risk", f"{high_risk}", f"{high_risk/total*100:.1f}%")
            
            # Detailed results table
            st.markdown("### 📊 Detailed Results")
            
            # Add color coding
            def color_risk(val):
                if val == 'No Risk':
                    return 'background-color: #d4edda'
                elif val == 'Low Risk (1-30 DPD)':
                    return 'background-color: #fff3cd'
                elif val == 'Medium Risk (31-60 DPD)':
                    return 'background-color: #ffe5b4'
                else:
                    return 'background-color: #f8d7da'
            
            styled_results = results.style.applymap(color_risk, subset=['Risk_Level'])
            st.dataframe(styled_results, use_container_width=True)
            
            # Download button
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="delinquency_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Risk distribution chart
            st.markdown("### 📈 Risk Distribution")
            st.bar_chart(results['Risk_Level'].value_counts())
            
            # High risk customers
            high_risk_customers = results[results['Predicted_DPD_Bucket'] >= 2]
            if len(high_risk_customers) > 0:
                st.markdown("### ⚠️ High-Priority Customers (Medium & High Risk)")
                st.dataframe(
                    high_risk_customers[['Customer_ID', 'Risk_Level', 'Prob_31-60_DPD', 'Prob_61+_DPD']].sort_values('Prob_61+_DPD', ascending=False),
                    use_container_width=True
                )
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.exception(e)
    
    else:
        st.info("👆 Please upload a file to begin prediction")
        
        # Show sample format
        with st.expander("ℹ️ Required File Format"):
            st.markdown("""
            Your file should contain the following columns:
            - **Customer_ID** or **Customer ID**: Unique customer identifier
            - **Credit_Limit** or **Credit Limit**: Credit card limit
            - **Utilisation_%** or **Utilisation %**: Credit utilization percentage
            - **Avg_Payment_Ratio** or **Avg Payment Ratio**: Average payment ratio
            - **Min_Due_Paid_Frequency** or **Min Due Paid Frequency**: Minimum due payment frequency
            - **Merchant_Mix_Index** or **Merchant Mix Index**: Merchant mix index
            - **Cash_Withdrawal_%** or **Cash Withdrawal %**: Cash withdrawal percentage
            - **Recent_Spend_Change_%** or **Recent Spend Change %**: Recent spending change percentage
            
            The file can be in Excel (.xlsx, .xls) or CSV format.
            """)


if __name__ == "__main__":
    main()
