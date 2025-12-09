# predict_app.py
# Run: streamlit run predict_app.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Credit Card Delinquency Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Stunning Dark Blue Gradient CSS
st.markdown("""
    <style>
    /* Import Professional Font */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    /* Remove Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Global Font */
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    
    /* Stunning Dark Blue Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        background-attachment: fixed;
    }
    
    /* Main Container with Glass Effect */
    .main .block-container {
        padding: 3rem 4rem;
        max-width: 1400px;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        margin: 2rem auto;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Header Container */
    .header-container {
        text-align: center;
        padding: 2rem 0 3rem 0;
        border-bottom: 3px solid #e8eef5;
        margin-bottom: 3rem;
    }
    
    /* Main Title with Gradient */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        letter-spacing: -0.5px;
    }
    
    /* Subtitle */
    .subtitle {
        font-size: 1.25rem;
        color: #64748b;
        font-weight: 400;
        margin-bottom: 0.5rem;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px 24px;
        border-radius: 25px;
        font-size: 0.95rem;
        font-weight: 500;
        margin-top: 1rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.6rem;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 2.5rem 0 1.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e8eef5;
    }
    
    /* Upload Box with Glow */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 3px dashed #667eea;
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        box-shadow: 0 0 30px rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        box-shadow: 0 0 40px rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }
    
    [data-testid="stFileUploader"] label {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        color: #1a365d !important;
    }
    
    /* Gradient Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.85rem 2.5rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.05rem;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
        transition: all 0.3s ease;
        width: 100%;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.7);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border: none;
        padding: 0.85rem 2.5rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.05rem;
        box-shadow: 0 8px 25px rgba(17, 153, 142, 0.5);
        transition: all 0.3s ease;
        width: 100%;
        letter-spacing: 0.5px;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(17, 153, 142, 0.7);
        background: linear-gradient(135deg, #38ef7d 0%, #11998e 100%);
    }
    
    /* Metric Cards Enhancement */
    [data-testid="stMetricValue"] {
        font-size: 2.8rem;
        font-weight: 700;
    }
    
    /* Success/Info Messages with Dark Theme */
    .stSuccess {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 5px solid #10b981;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        color: #065f46;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
    }
    
    .stInfo {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 5px solid #3b82f6;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        color: #1e40af;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 5px solid #f59e0b;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        color: #92400e;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.2);
    }
    
    /* Expander with Gradient */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8fafc 0%, #e8eef5 100%);
        border-radius: 10px;
        font-weight: 600;
        color: #1a365d;
        padding: 1rem 1.5rem;
        border: 1px solid #e8eef5;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #e8eef5 0%, #cbd5e1 100%);
    }
    
    /* Data Table */
    .dataframe {
        font-size: 0.95rem;
        border: 1px solid #e8eef5;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Remove Clutter */
    [data-testid="stToolbar"] {display: none;}
    .css-1dp5vir {display: none;}
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
    data.columns = data.columns.str.strip()
    
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
    
    data.rename(columns=column_mapping, inplace=True)
    
    if 'DPD_Bucket_Next_Month' in data.columns:
        data = data.drop('DPD_Bucket_Next_Month', axis=1)
    
    customer_ids = data['Customer_ID'].copy()
    data = data.drop('Customer_ID', axis=1)
    
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
    features_array = features.to_numpy()
    features_scaled = scaler.transform(features_array)
    predictions = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)
    
    results = pd.DataFrame({
        'Customer_ID': customer_ids.values,
        'Predicted_DPD_Bucket': predictions,
    })
    
    results['Risk_Level'] = results['Predicted_DPD_Bucket'].map({
        0: 'No Risk',
        1: 'Low Risk (1-30 DPD)',
        2: 'Medium Risk (31-60 DPD)',
        3: 'High Risk (61+ DPD)'
    })
    
    results['Prob_No_Risk'] = probabilities[:, 0].round(4)
    results['Prob_1-30_DPD'] = probabilities[:, 1].round(4)
    results['Prob_31-60_DPD'] = probabilities[:, 2].round(4)
    results['Prob_61+_DPD'] = probabilities[:, 3].round(4)
    
    return results

def main():
    # Stunning Header
    st.markdown("""
    <div class="header-container">
        <div style="font-size: 4rem; margin-bottom: 1rem; filter: drop-shadow(0 0 20px rgba(102, 126, 234, 0.5));">💳</div>
        <h1 class="main-title">Credit Card Delinquency Predictor</h1>
        <p class="subtitle">AI-Powered Risk Assessment System for Financial Institutions</p>
        <span class="badge">✨ Powered by Advanced Machine Learning</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, scaler = load_model_and_scaler()
    
    # Upload Section
    st.markdown('<div class="section-header">📤 Upload Customer Data</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        uploaded_file = st.file_uploader(
            "Upload your Excel or CSV file",
            type=['xlsx', 'xls', 'csv'],
            help="Drag and drop your file here or click to browse"
        )
    
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                try:
                    data = pd.read_excel(uploaded_file, sheet_name='Sample')
                except:
                    data = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Successfully loaded **{data.shape[0]} customers** with **{data.shape[1]} attributes**")
            
            # Preview
            with st.expander("👁️ View Data Preview"):
                st.dataframe(data.head(10), use_container_width=True)
            
            # Process
            with st.spinner("🔄 Processing data and generating predictions..."):
                features, customer_ids = preprocess_data(data)
                results = make_predictions(model, scaler, features, customer_ids)
            
            st.balloons()
            
            # Results
            st.markdown('<div class="section-header">📊 Risk Assessment Results</div>', unsafe_allow_html=True)
            
            # Stunning Metric Cards
            risk_counts = results['Risk_Level'].value_counts()
            total = len(results)
            
            col1, col2, col3, col4 = st.columns(4)
            
            metrics = [
                ("🟢 No Risk", "#10b981", "#ecfdf5", "#d1fae5", risk_counts.get('No Risk', 0)),
                ("🟡 Low Risk", "#f59e0b", "#fffbeb", "#fef3c7", risk_counts.get('Low Risk (1-30 DPD)', 0)),
                ("🟠 Medium Risk", "#f97316", "#fff7ed", "#ffedd5", risk_counts.get('Medium Risk (31-60 DPD)', 0)),
                ("🔴 High Risk", "#ef4444", "#fef2f2", "#fee2e2", risk_counts.get('High Risk (61+ DPD)', 0))
            ]
            
            for col, (name, color, bg1, bg2, count) in zip([col1, col2, col3, col4], metrics):
                with col:
                    percentage = (count/total*100) if total > 0 else 0
                    st.markdown(f"""
                    <div style='
                        background: linear-gradient(135deg, {bg1} 0%, {bg2} 100%);
                        padding: 2rem 1rem;
                        border-radius: 15px;
                        text-align: center;
                        border: 3px solid {color};
                        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1), 0 0 30px {color}40;
                        transition: transform 0.3s ease;
                    '>
                        <h3 style='color: {color}; margin: 0 0 1rem 0; font-size: 1.1rem; font-weight: 600;'>{name}</h3>
                        <h1 style='color: {color}; margin: 0.5rem 0; font-size: 3.2rem; font-weight: 700; text-shadow: 0 2px 10px {color}40;'>{count}</h1>
                        <div style='background: white; padding: 8px; border-radius: 10px; margin-top: 1rem; border: 2px solid {color};'>
                            <p style='color: {color}; margin: 0; font-size: 1.3rem; font-weight: 700;'>{percentage:.1f}%</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Charts
            st.markdown('<div class="section-header">📈 Visual Analytics</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_pie = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    title="<b>Risk Distribution by Category</b>",
                    color=risk_counts.index,
                    color_discrete_map={
                        'No Risk': '#10b981',
                        'Low Risk (1-30 DPD)': '#f59e0b',
                        'Medium Risk (31-60 DPD)': '#f97316',
                        'High Risk (61+ DPD)': '#ef4444'
                    },
                    hole=0.4
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label', textfont_size=14, textfont_color='white')
                fig_pie.update_layout(height=400, showlegend=True, title_font_size=18, title_font_color='#1a365d')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                fig_bar = px.bar(
                    x=risk_counts.index,
                    y=risk_counts.values,
                    title="<b>Customer Count by Risk Level</b>",
                    labels={'x': 'Risk Level', 'y': 'Number of Customers'},
                    color=risk_counts.index,
                    color_discrete_map={
                        'No Risk': '#10b981',
                        'Low Risk (1-30 DPD)': '#f59e0b',
                        'Medium Risk (31-60 DPD)': '#f97316',
                        'High Risk (61+ DPD)': '#ef4444'
                    }
                )
                fig_bar.update_layout(height=400, showlegend=False, title_font_size=18, title_font_color='#1a365d')
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Table
            st.markdown('<div class="section-header">📋 Detailed Customer Analysis</div>', unsafe_allow_html=True)
            
            def color_risk(val):
                colors = {
                    'No Risk': 'background-color: #10b981; color: white; font-weight: 600; padding: 10px; border-radius: 8px;',
                    'Low Risk (1-30 DPD)': 'background-color: #f59e0b; color: white; font-weight: 600; padding: 10px; border-radius: 8px;',
                    'Medium Risk (31-60 DPD)': 'background-color: #f97316; color: white; font-weight: 600; padding: 10px; border-radius: 8px;',
                    'High Risk (61+ DPD)': 'background-color: #ef4444; color: white; font-weight: 600; padding: 10px; border-radius: 8px;'
                }
                return colors.get(val, '')
            
            styled_results = results.style.applymap(color_risk, subset=['Risk_Level'])
            st.dataframe(styled_results, use_container_width=True, height=400)
            
            # Download
            csv = results.to_csv(index=False).encode('utf-8')
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label="📥 Download Complete Report (CSV)",
                    data=csv,
                    file_name=f"risk_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # High Priority
            high_risk_customers = results[results['Predicted_DPD_Bucket'] >= 2]
            if len(high_risk_customers) > 0:
                st.markdown('<div class="section-header">⚠️ High-Priority Customers</div>', unsafe_allow_html=True)
                st.warning(f"🚨 **{len(high_risk_customers)} customers** require immediate attention")
                st.dataframe(
                    high_risk_customers[['Customer_ID', 'Risk_Level', 'Prob_31-60_DPD', 'Prob_61+_DPD']].sort_values('Prob_61+_DPD', ascending=False).head(10),
                    use_container_width=True
                )
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    else:
        # Welcome
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.info("👆 **Please upload a customer data file to begin risk assessment**")
            
            st.markdown("""
            <div style='background: linear-gradient(135deg, #f8fafc 0%, #e8eef5 100%); padding: 2.5rem; border-radius: 15px; margin-top: 2rem; border: 1px solid #e8eef5; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);'>
                <h3 style='color: #1a365d; margin-bottom: 1.5rem; text-align: center;'>✨ What you'll get:</h3>
                <ul style='color: #64748b; font-size: 1.1rem; line-height: 2.2; list-style: none; padding: 0;'>
                    <li>⚡ <strong>Instant</strong> AI-powered risk predictions</li>
                    <li>📊 <strong>Interactive</strong> visual analytics</li>
                    <li>📥 <strong>Downloadable</strong> CSV reports</li>
                    <li>🎯 <strong>Priority</strong> customer identification</li>
                    <li>📈 <strong>Comprehensive</strong> portfolio analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with st.expander("📋 Required Data Format"):
            st.markdown("""
            Your file should contain these columns:
            
            | Column | Description |
            |--------|-------------|
            | **Customer_ID** | Unique customer identifier |
            | **Credit_Limit** | Credit card limit amount |
            | **Utilisation_%** | Percentage of credit utilized |
            | **Avg_Payment_Ratio** | Average payment ratio |
            | **Min_Due_Paid_Frequency** | Minimum payment frequency |
            | **Merchant_Mix_Index** | Spending diversification (0-1) |
            | **Cash_Withdrawal_%** | Cash withdrawal percentage |
            | **Recent_Spend_Change_%** | Spending change |
            
            **Supported formats:** Excel (.xlsx, .xls) or CSV files
            """)

if __name__ == "__main__":
    main()
