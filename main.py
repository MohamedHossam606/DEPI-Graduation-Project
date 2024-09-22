import streamlit as st
import pandas as pd
import numpy  as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import requests
from streamlit_lottie import st_lottie

Transactions_df = pd.read_csv("Notebooks/Preprocessing Notebooks/Final Sheets/Transactions.csv")

Q3_2017_Prediction = pd.read_csv("Notebooks/Preprocessing Notebooks/Final Sheets/Q2_features 2017 Q3 Purchases Prediction.csv")
Q4_2017_Prediction = pd.read_csv("Notebooks/Preprocessing Notebooks/Final Sheets/Q3_features 2017 Q4 Purchases Prediction.csv")
Q1_2018_Prediction = pd.read_csv("Notebooks/Preprocessing Notebooks/Final Sheets/Q4_features 2018 Q1 Purchases Prediction.csv")

# Set page config
st.set_page_config(page_title="Purchases Prediction",initial_sidebar_state='expanded')

# Load the Lottie animation from the URL
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

animation = load_lottieurl('https://lottie.host/62653c16-c2aa-4aa6-8c95-d67b3cdbfd36/ucEPWXcKix.json')

col1, col2 = st.columns([3, 1])
with col1:
    st.title('90-Day Purchases Prediction')
with col2:
    if animation is not None:
        st_lottie(animation, speed=0.95, quality='high', width=180, height=180)
    else:
        st.error("Animation failed to load.")


tab1,tab2 = st.tabs(["Purchases Prediction", "Process"])
with tab1:
    st.markdown('''
    #### **Objectives:**
    ##### **Predict Customer Spending Behavior:**
    * Develop a model to predict whether a customer will make a purchase within the next 90 days based on their historical transaction data and customer demographics.
    * Help the business prioritize customer retention and marketing efforts.
                
    ##### **Enhance Customer Retention:**
    * Identify at-risk customers who are unlikely to make a purchase in the near future.
    
    ##### **Optimize Marketing and Resource Allocation:**
    * Use the model’s predictions to focus marketing resources on high-probability spenders for personalized promotions.
    ''')

with tab2:
    st.markdown('''
    #### **Process:**
    ##### **1. Data Collection & Preprocessing:**
    * **Data Sources:** Collect transaction data (online and offline purchases), customer demographic information, and behavioral attributes.
    * **Cleaning:** Remove irrelevant records (e.g., canceled orders, deceased customers) and handle missing values (e.g., filling job titles, removing unnecessary columns).
    * **Feature Engineering:** Generate features such as total spend, transaction frequency, recency of purchases, and customer demographics (e.g., age, job industry, wealth segment).
    * **Handling Outliers:** Remove outliers from numerical columns to prevent skewing the model’s performance.
    
    ##### **2. Model Training:**
    * **Binary Classification:** Train a model to predict whether a customer will make a purchase within the next 90 days (binary target).
    * **Data Splitting:** Split the dataset into training and testing sets by grouping the transactions into quarterly datasets (Q1, Q2, Q3).
    * **Preprocessing:** Apply scaling to numerical features and one-hot encoding to categorical features to prepare the data for machine learning algorithms.
    * **Model Selection:** Use Random Forest as the primary classification model, fine-tuning hyperparameters through grid search and cross-validation.
    * **Prediction: Generate** predicted probabilities for the next 90-day spending likelihood.
    
    ##### **3. Model Evaluation:**
    * **Metrics:** Evaluate model performance using accuracy, ROC-AUC, and precision-recall metrics to assess the effectiveness of predicting customer spending.
    ''')

    if st.button('Show Transactions Sample Data') :
        st.dataframe(Transactions_df.sample(10),use_container_width=True)

st.divider()

quarter_files = {
    '2017 Q3 Prediction': Q3_2017_Prediction,
    '2017 Q4 Prediction': Q4_2017_Prediction,
    '2018 Q1 Prediction': Q1_2018_Prediction
}

selected = st.selectbox('Select Quarter Prediction to filter by:', list(quarter_files.keys()), index=2)
selected_df = quarter_files[selected]

# Define the color palette
colors = ['#606C38', '#283618', '#FEFAE0', '#DDA15E', '#BC6C25']

if selected == "2017 Q3 Prediction":
    accuracy = 76.8
elif selected == "2017 Q4 Prediction":
    accuracy = 76.7
else:
    accuracy = 76


# Display prediction accuracy gauge chart
st.write("### Model Prediction Accuracy")
st.write("This gauge shows the model's prediction accuracy for the selected quarter. A higher value indicates better predictive performance.")
fig0 = go.Figure(go.Indicator(
    mode="gauge+number",
    value=accuracy,
    number={'suffix': "%"},
    title={'text': "Prediction Accuracy"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "#606C38"},
        'steps': [
            {'range': [0, 50], 'color': "#FEFAE0"},
            {'range': [50, 80], 'color': "#283618"},
            {'range': [80, 100], 'color': "#DDA15E"}],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 77}}))
st.plotly_chart(fig0)
st.divider()

# Histogram of predicted probabilities
st.write("### Distribution of Predicted Probabilities")
st.write("This histogram shows the distribution of predicted probabilities for customers likely to spend in the next 90 days. A higher concentration near 100% indicates strong predictions of spending.")
fig1 = px.histogram(selected_df, 
                    x='spend_next_90_days_pred_proba', 
                    nbins=20, 
                    title='Distribution of Predicted Probabilities for Spending in Next 90 Days',
                    labels={'spend_next_90_days_pred_proba': 'Predicted Probability (%)'},
                    color_discrete_sequence=[colors[0]])
fig1.update_layout(bargap=0.1, xaxis_title="Probability (%)", yaxis_title="Count")
fig1.update_layout(
    plot_bgcolor=colors[2],
    title_font=dict(size=20, color=colors[0]),
    xaxis_title_font=dict(size=16, color=colors[0]),
    yaxis_title_font=dict(size=16, color=colors[0])
)
st.plotly_chart(fig1)
st.divider()

# Pie chart of predicted spending behavior
st.write("### Predicted Spending Behavior")
st.write("This pie chart shows the proportion of customers predicted to spend (1) and not to spend (0) in the next 90 days. It helps businesses understand the overall expected spending behavior.")
fig2 = px.pie(selected_df, 
              names='spend_next_90_days_pred', 
              title='Proportion of Predicted Spending Behavior',
              labels={'spend_next_90_days_pred': 'Predicted Spending'},
              color_discrete_sequence=colors[:2])
st.plotly_chart(fig2)
st.divider()

# Bar charts of state and wealth segment
col1, col2 = st.columns(2)

with col1:
    st.write("### Predicted Spend Probability by State")
    st.write("This bar chart shows the average predicted spend probability by state. It helps identify geographical regions with high or low expected spending.")
    fig3 = px.bar(Q3_2017_Prediction, 
                  x='state', 
                  y='spend_next_90_days_pred_proba', 
                  title='Predicted Spend Probability by State',
                  labels={'spend_next_90_days_pred_proba': 'Avg Predicted Spend Probability (%)'},
                  color_discrete_sequence=colors)
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.write("### Predicted Spend Probability by Wealth Segment")
    st.write("This bar chart shows the average predicted spend probability by customer wealth segment. It highlights which wealth segments are more likely to spend.")
    fig4 = px.bar(Q3_2017_Prediction, 
                  x='wealth_segment', 
                  y='spend_next_90_days_pred_proba', 
                  title='Predicted Spend Probability by Wealth Segment',
                  labels={'spend_next_90_days_pred_proba': 'Avg Predicted Spend Probability (%)'},
                  color_discrete_sequence=colors)
    st.plotly_chart(fig4, use_container_width=True)
st.divider()

# Confusion matrix and ROC curve (if not 2018 Q1)
if selected != "2018 Q1 Prediction":
    st.write("### Confusion Matrix")
    st.write("The confusion matrix shows how well the model is performing in predicting customer spending (True Positive and True Negative rates). A higher diagonal value indicates better performance.")
    conf_matrix = confusion_matrix(selected_df['spend_next_90_days'], selected_df['spend_next_90_days_pred'])
    fig5 = ff.create_annotated_heatmap(conf_matrix, 
                                       x=['Predicted 0', 'Predicted 1'], 
                                       y=['Actual 0', 'Actual 1'], 
                                       colorscale=[[0.0, colors[1]], [1.0, colors[4]]], 
                                       showscale=True)
    st.plotly_chart(fig5)
    st.divider()

    # Calculate ROC curve
    st.write("### ROC Curve")
    st.write("The ROC curve shows the trade-off between sensitivity (true positive rate) and specificity (false positive rate). A model with a curve closer to the top left corner has better predictive power.")
    fpr, tpr, _ = roc_curve(selected_df['spend_next_90_days'], selected_df['spend_next_90_days_pred_proba'])
    # Create the ROC curve plot
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve', line=dict(color=colors[0], width=3)))
    fig6.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Model', line=dict(color=colors[4], dash='dash')))
    # Customize layout
    fig6.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        plot_bgcolor=colors[2],
        title_font=dict(size=20, color=colors[0]),
        xaxis_title_font=dict(size=16, color=colors[0]),
        yaxis_title_font=dict(size=16, color=colors[0])
    )
    st.plotly_chart(fig6)
    st.divider()
else:
    # User input for new customer prediction
    st.write("## Predict New Customer Behavior")
    st.write("Enter customer details below to predict the likelihood of them making a purchase within the next 90 days:")

    @st.cache_resource
    def load_model():
        return joblib.load('Notebooks/Saved Models/Q3 tranied pipeline_model.pkl')

    model_pipeline = load_model()

    col1, col2 = st.columns(2)

    with col1:
        past_3_years_bike_purchases = st.number_input('Past 3 Years Bike Purchases', min_value=0, max_value=100, value=0, help="Number of bikes purchased by the customer in the last 3 years.")
        age = st.number_input('Age', min_value=14, max_value=100)
        job_industry_category = st.selectbox('Job Industry Category', ['N/A', 'Manufacturing', 'Financial Services', 'Health', 'Retail', 'Property', 'IT', 'Entertainment', 'Argiculture', 'Telecommunications'])
        wealth_segment = st.selectbox('Wealth Segment', ['Mass Customer', 'Affluent Customer', 'High Net Worth'])
        owns_car = st.selectbox('Owns Car', ['Yes', 'No'])

    with col2:
        tenure = st.number_input('Tenure', min_value=0, max_value=40, help="Months of customer loyalty")
        property_valuation = st.number_input('Property Valuation', min_value=1, max_value=12)
        total_spend = st.number_input('Total Spend (last 3 months)', min_value=0.0, help="Customer's total spend in the last 3 months")
        transaction_count = st.number_input('Transaction Count (last 3 months)', min_value=0, help="Number of transaction customer's made in the last 3 months")
        recency = st.number_input('Recency (days since last purchase)', min_value=0, help="Number of days since the customer's last purchase")

    state = st.selectbox('State', ['New South Wales', 'Victoria', 'Queensland'])

    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'past_3_years_bike_related_purchases': [past_3_years_bike_purchases],
        'age': [age],
        'job_industry_category': [job_industry_category],
        'wealth_segment': [wealth_segment],
        'owns_car': [owns_car],
        'tenure': [tenure],
        'state': [state],
        'property_valuation': [property_valuation],
        'total_spend': [total_spend],
        'transaction_count': [transaction_count],
        'recency': [recency]
    })

    st.write("Customer Data Preview:")
    st.dataframe(input_data.style.hide(axis="index"))

    if st.button('Predict'):
        # Preprocess and predict
        prediction_proba = model_pipeline.predict_proba(input_data)[:, 1]  # Probabilities
        prediction = model_pipeline.predict(input_data)  # Predictions (0 or 1)

        if prediction[0] == 1:
            st.success(f"Prediction: This customer is likely to spend within the next 90 days with a probability of {prediction_proba[0] * 100:.1f}%.")
        else:
            st.warning(f"Prediction: This customer is unlikely to spend within the next 90 days with a probability of {prediction_proba[0] * 100:.1f}%.")
