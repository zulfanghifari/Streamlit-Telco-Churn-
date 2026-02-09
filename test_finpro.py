import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load the model and LIME explainer
@st.cache_resource
def load_model():
    with open('customer-churn-telco-prediction.sav', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_lime_explainer():
    with open('lime_explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)
    return explainer

# Load model and explainer
model = load_model()
lime_explainer = load_lime_explainer()

# Title
st.title("📊 Customer Churn Prediction System")
st.markdown("---")

# Create input form in the middle of the screen
col_left, col_middle, col_right = st.columns([1, 3, 1])

with col_middle:
    st.header("Customer Information")
    
    # Group 1: Demographics
    st.subheader("👤 Demographics")
    demo_col1, demo_col2, demo_col3 = st.columns(3)
    
    with demo_col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
    
    with demo_col2:
        seniorcitizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        dependents = st.selectbox("Dependents", ["Yes", "No"])
    
    with demo_col3:
        partner = st.selectbox("Partner", ["Yes", "No"])
        population = st.number_input("Population", min_value=0, value=1000, step=100)
    
    st.markdown("---")
    
    # Group 2: Service Information
    st.subheader("📞 Service Information")
    service_col1, service_col2, service_col3 = st.columns(3)
    
    with service_col1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12, step=1)
        contract = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
        phoneservice = st.selectbox("Phone Service", ["Yes", "No"])
    
    with service_col2:
        multiplelines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internetservice = st.selectbox("Internet Service", ["DSL", "Fiber Optic", "No"])
        onlinesecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    
    with service_col3:
        onlinebackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        deviceprotection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        techsupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    
    st.markdown("---")
    
    # Group 3: Streaming & Entertainment
    st.subheader("📺 Streaming & Entertainment")
    stream_col1, stream_col2, stream_col3 = st.columns(3)
    
    with stream_col1:
        streamingtv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    
    with stream_col2:
        streamingmovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    
    with stream_col3:
        streamingmusic = st.selectbox("Streaming Music", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    st.markdown("---")
    
    # Group 4: Billing & Payment
    st.subheader("💳 Billing & Payment")
    billing_col1, billing_col2, billing_col3 = st.columns(3)
    
    with billing_col1:
        monthlycharges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=70.0, step=0.5)
        paymentmethod = st.selectbox("Payment Method", [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ])
    
    with billing_col2:
        paperlessbilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        unlimiteddata = st.selectbox("Unlimited Data", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    with billing_col3:
        avgmonthlylongdistancecharges = st.number_input(
            "Avg Monthly Long Distance Charges ($)", 
            min_value=0.0, 
            max_value=100.0, 
            value=10.0, 
            step=0.5
        )
        avgmonthlygbdownload = st.number_input(
            "Avg Monthly GB Download", 
            min_value=0, 
            max_value=1000, 
            value=20, 
            step=1
        )
    
    st.markdown("---")
    
    # Group 5: Customer Value & Satisfaction
    st.subheader("⭐ Customer Value & Satisfaction")
    value_col1, value_col2, value_col3 = st.columns(3)
    
    with value_col1:
        cltv = st.number_input("Customer Lifetime Value", min_value=0, max_value=10000, value=3000, step=100)
        satisfactionscore = st.number_input("Satisfaction Score", min_value=1, max_value=5, value=3, step=1)
    
    with value_col2:
        numberofreferrals = st.number_input("Number of Referrals", min_value=0, max_value=20, value=0, step=1)
        referredafriend = st.selectbox("Referred a Friend", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    st.markdown("---")
    
    # Predict button
    predict_button = st.button("🔮 Predict Churn", type="primary", use_container_width=True)
    
    if predict_button:
        # Create input dataframe
        input_data = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'seniorcitizen': [seniorcitizen],
            'dependents': [dependents],
            'partner': [partner],
            'population': [population],
            'tenure': [tenure],
            'contract': [contract],
            'phoneservice': [phoneservice],
            'multiplelines': [multiplelines],
            'internetservice': [internetservice],
            'onlinesecurity': [onlinesecurity],
            'onlinebackup': [onlinebackup],
            'deviceprotection': [deviceprotection],
            'techsupport': [techsupport],
            'streamingtv': [streamingtv],
            'streamingmovies': [streamingmovies],
            'streamingmusic': [streamingmusic],
            'monthlycharges': [monthlycharges],
            'paymentmethod': [paymentmethod],
            'paperlessbilling': [paperlessbilling],
            'unlimiteddata': [unlimiteddata],
            'avgmonthlylongdistancecharges': [avgmonthlylongdistancecharges],
            'avgmonthlygbdownload': [avgmonthlygbdownload],
            'cltv': [cltv],
            'satisfactionscore': [satisfactionscore],
            'numberofreferrals': [numberofreferrals],
            'referredafriend': [referredafriend]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Display prediction
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            st.metric(
                label="Churn Prediction",
                value="Will Churn" if prediction == 1 else "Will Stay",
                delta="High Risk" if prediction == 1 else "Low Risk",
                delta_color="inverse"
            )
        
        with result_col2:
            st.metric(
                label="Churn Probability",
                value=f"{prediction_proba[1]:.1%}"
            )
        
        with result_col3:
            st.metric(
                label="Retention Probability",
                value=f"{prediction_proba[0]:.1%}"
            )
        
        # Progress bar for churn probability
        st.progress(prediction_proba[1])
        
        # LIME Explanation
        st.markdown("---")
        st.subheader("🔍 LIME Explanation")
        st.write("Understanding the factors influencing this prediction:")
        
        with st.spinner("Generating explanation..."):
            try:
                # CRITICAL: Get the preprocessor from the pipeline
                preprocessor = model.named_steps['preprocessor']
                
                # Transform the input data using the preprocessor
                # Since set_output(transform="pandas") is used, this will return a DataFrame
                preprocessed_data = preprocessor.transform(input_data)
                
                # Convert to numpy array for LIME (LIME expects numpy arrays)
                # We need to extract the values from the DataFrame
                preprocessed_array = preprocessed_data
                
                # Get feature names after preprocessing
                feature_names_after_preprocessing = preprocessor.get_feature_names_out().tolist()
                
                # Create a prediction function that works with preprocessed data
                def predict_fn(preprocessed_array):
                    # Create DataFrame from the preprocessed array
                    preprocessed_df = pd.DataFrame(
                        preprocessed_array, 
                        columns=feature_names_after_preprocessing
                    )
                    
                    # Get the remaining steps in the pipeline (after preprocessor)
                    # We need to apply sampler, feature_selector, and classifier
                    remaining_pipeline = model[1:]  # This gets all steps after preprocessor
                    
                    # Predict using the remaining pipeline
                    return remaining_pipeline.predict_proba(preprocessed_df)
                
                # Generate LIME explanation
                explanation = lime_explainer.explain_instance(
                    preprocessed_array[0],
                    predict_fn,
                    num_features=10
                )
                
                # Plot the explanation
                fig = explanation.as_pyplot_figure()
                fig.set_size_inches(12, 6)
                st.pyplot(fig)
                plt.close()
                
                # Additional explanation details
                with st.expander("📋 Detailed Feature Contributions"):
                    explanation_list = explanation.as_list()
                    
                    contrib_df = pd.DataFrame(explanation_list, columns=['Feature', 'Contribution'])
                    contrib_df['Impact'] = contrib_df['Contribution'].apply(
                        lambda x: '🔴 Increases Churn' if x > 0 else '🟢 Decreases Churn'
                    )
                    contrib_df['Abs_Contribution'] = contrib_df['Contribution'].abs()
                    contrib_df = contrib_df.sort_values('Abs_Contribution', ascending=False)
                    
                    st.dataframe(
                        contrib_df[['Feature', 'Contribution', 'Impact']],
                        use_container_width=True,
                        hide_index=True
                    )
                
            except Exception as e:
                st.error(f"Error generating LIME explanation: {str(e)}")
                st.write("Debug information:")
                st.write(f"Error type: {type(e).__name__}")
                st.write(f"Preprocessed data type: {type(preprocessed_data)}")
                if isinstance(preprocessed_data, pd.DataFrame):
                    st.write(f"Preprocessed data shape: {preprocessed_data.shape}")
                    st.write(f"Feature names: {preprocessor.get_feature_names_out().tolist()[:10]}...")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Customer Churn Prediction System | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)   