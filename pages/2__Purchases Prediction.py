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
import os

# Get the current directory of your script
current_directory = os.path.dirname(os.path.abspath(__file__))

Transactions_df = pd.read_csv(os.path.join(current_directory, "../Notebooks/1_Preprocessing Notebooks/Final Sheets/transactions_details.csv"))

Q3_2017_Prediction = pd.read_csv(os.path.join(current_directory,"../Notebooks/3_Prediction Notebooks/Output Sheets/Q2_features 2017 Q3 Purchases Prediction.csv"))
Q4_2017_Prediction = pd.read_csv(os.path.join(current_directory,"../Notebooks/3_Prediction Notebooks/Output Sheets/Q3_features 2017 Q4 Purchases Prediction.csv"))
Q1_2018_Prediction = pd.read_csv(os.path.join(current_directory,"../Notebooks/3_Prediction Notebooks/Output Sheets/Q4_features 2018 Q1 Purchases Prediction.csv"))

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(current_directory,'../Notebooks/3_Prediction Notebooks/Saved Models/Q3 tranied pipeline_model.pkl'))

st.set_page_config(page_title="Purchases Prediction",initial_sidebar_state='expanded')

# Load the Lottie animation from the URL
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

animation = load_lottieurl('https://lottie.host/52763640-3a89-43c3-b466-b8d5791f8cda/xykMp0iuj1.json')

col1, col2 = st.columns([2, 1])
with col1:
    st.title('90-Day Purchases Prediction')
with col2:
    if animation is not None:
        st_lottie(animation, speed=0.95, quality='high', width=180, height=180)
    else:
        st.error("Animation failed to load.")

st.image("pages/assets/2_Purchases Prediction/Prediction.jpg")
tab1,tab2 = st.tabs(["Purchases Prediction", "Process"])
with tab1:
    st.markdown('''
    ### **Objectives:**
    ##### **Customer Purchase Prediction:**
    * The primary objective is to predict whether a customer will make a purchase in the next 90 days based on their transaction history and customer attributes.
    * Help the business prioritize customer retention and marketing efforts.
                
    ##### **Customer Segmentation:**
    * Use the model to segment customers into those who are likely to make a purchase and those who are not, helping to create more focused marketing efforts.
    
    ### **How the Model Could Be Used:**
    ##### **Marketing Campaigns:**
    * The prediction of a customer's likelihood to purchase can be used to design tailored marketing campaigns (e.g., promotional offers for customers with a low probability of purchase).
    
    ##### **Revenue Forecasting:**
    * : The model helps in forecasting potential revenue from recurring customers in the next quarter by identifying high-value customers likely to make large purchases.
    
    ##### **Inventory Management:**
    * By forecasting demand, the store can optimize stock levels based on the expected number of customers making purchases in the next quarter.
   
    ''')

with tab2:
    st.markdown('''
    #### **Process to Build the Model:**
    ##### **1. Data Preprocessing:**
    * **Cleaning:** Merged customer and transaction datasets and cleaned the data by removing unnecessary columns (e.g., deceased customers and cancelled orders) & Dealt with missing values.
    * **Feature Engineering:** Generate features such as total spend, transaction frequency, recency of purchases, and customer demographics (e.g., age, job industry, wealth segment).
    * **Handling Outliers:** Remove outliers from numerical columns to prevent skewing the model’s performance.
    
    ##### **1. Data Splitting:**
    * Split the transaction data into quarters to simulate a time-based holdout set, ensuring that your model predicts future behavior rather than learning from future data.
    ''')

    st.image("pages/assets/2_Purchases Prediction/year-quarters.png")

    st.markdown('''
    * For Example: **Q3 Data (Training Set):** used transactions from Q3 (July–September). 2017 to generate the training features and the target variable. The target here is whether the customer made a purchase in Q4 (October–December). 
    * **Q4 Data (Testing Set):** used transactions from Q4 2017 to generate the test features, and the target was whether the customer made a purchase in Q1 2018
    
    ##### **2. Model Training:**
    * **Binary Classification:** Train a model to predict whether a customer will make a purchase within the next 90 days (binary target).
    * **Preprocessing Pipeline:** Used RobustScaler for scaling numerical features and OneHotEncoder for encoding categorical features, Combined them into a single preprocessing pipeline using ColumnTransformer. 
    * **Model Selection:** Trained several models with hyperparameter tuning using GridSearchCV and evaluating performance using cross-validation, including:(Logistic Regression, Random Forest, Support Vector Machine)
    
    ##### **3. Model Evaluation:**
    * **Metrics:** Evaluate model performance using accuracy, confusion matrix, and classification report to understand its precision, recall, and F1-score.
    * **Best Mode:** Random Forest was chosen as the final model based on performance AVG. accuracy = 76%.
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
selected_df['spend_next_90_days_pred_C'] = selected_df['spend_next_90_days_pred'].astype(str)

colors = ['#386641', '#6A994E', '#A7C957', '#B7E4C7', '#2F5233']

if selected == "2017 Q3 Prediction":
    accuracy = 77
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
        'bar': {'color': "#6A994E"},
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 77}}))
st.plotly_chart(fig0)
st.divider()


# --- Histogram: Distribution of Predicted Probabilities ---
st.write("### Distribution of Predicted Probabilities")
st.write("This histogram shows the distribution of predicted probabilities for customers likely to spend in the next 90 days. A higher concentration near 100% indicates strong predictions of spending.")

fig1 = px.histogram(selected_df, 
                    x='spend_next_90_days_pred_proba', 
                    nbins=20, 
                    title='Distribution of Predicted Probabilities for Spending in Next 90 Days',
                    labels={'spend_next_90_days_pred_proba': 'Predicted Probability (%)'},
                    color_discrete_sequence=[colors[0]])
fig1.update_layout(bargap=0.1, xaxis_title="Probability (%)", yaxis_title="Count")
st.plotly_chart(fig1)
st.divider()


# --- Pie Chart: Predicted Spending Behavior ---
st.write("### Predicted Spending Behavior")
st.write("This pie chart shows the proportion of customers predicted to spend (1) and not spend (0) in the next 90 days. It helps businesses understand the overall expected spending behavior.")

st.write("### Filter by State")
all_states = selected_df['state'].unique()
state_filter = st.selectbox("Select a State", options=["All"] + list(all_states), index=0)

# Filter the dataframe based on the selected state
filtered_df = selected_df if state_filter == "All" else selected_df[selected_df['state'] == state_filter]

# Calculate the total number of customers who will spend and will not spend
will_spend_count = filtered_df[filtered_df['spend_next_90_days_pred'] == 1].shape[0]
will_not_spend_count = filtered_df[filtered_df['spend_next_90_days_pred'] == 0].shape[0]

col1, col2 = st.columns(2)
# Pie chart for predicted spending behavior
with col1:
    fig2 = px.pie(filtered_df, 
                  names='spend_next_90_days_pred_C', 
                  title=f'Proportion of Predicted Spending Behavior ({state_filter})',
                  labels={0: 'Will Not Spend', 1: 'Will Spend'},
                  color_discrete_sequence=colors[:2],
                  hole=0.3)
    fig2.update_traces(textinfo='percent+label', showlegend=True)
    st.plotly_chart(fig2)

# Display metrics for the selected state
with col2:
    st.write(" ")
    st.write(" ")
    st.metric(f"Total Customers Who Will Spend ({state_filter})", value=f"{will_spend_count:,}")
    st.metric(f"Total Customers Who Will Not Spend ({state_filter})", value=f"{will_not_spend_count:,}")
st.divider()


# --- Bar Charts: Spend Predictions by State and Wealth Segment ---
col1, col2 = st.columns(2)
# Bar chart for number of customers by state who will spend/not spend
with col1:
    st.write("### Spend Predictions by State")
    st.write("This bar chart shows the count of customers by state for those predicted to spend or not spend in the next 90 days.")
    
    # Group by state and spend_next_90_days_pred, then count
    state_counts = selected_df.groupby(['state', 'spend_next_90_days_pred_C']).size().reset_index(name='customer_count')
    
    fig3 = px.bar(state_counts, 
                  x='state', 
                  y='customer_count', 
                  color='spend_next_90_days_pred_C', 
                  barmode='group',
                  title='Customer Spend Predictions by State',
                  labels={'customer_count': 'Number of Customers'},
                  color_discrete_sequence=colors)
    st.plotly_chart(fig3, use_container_width=True)

# Bar chart for number of customers by wealth segment who will spend/not spend
with col2:
    st.write("### Spend Predictions by Wealth Segment")
    st.write("This bar chart shows the count of customers by wealth segment for those predicted to spend or not spend in the next 90 days.")
    
    wealth_segment_counts = selected_df.groupby(['wealth_segment', 'spend_next_90_days_pred_C']).size().reset_index(name='customer_count')
    
    fig4 = px.bar(wealth_segment_counts, 
                  x='wealth_segment', 
                  y='customer_count', 
                  color='spend_next_90_days_pred_C', 
                  barmode='group',
                  title='Customer Spend Predictions by Wealth Segment',
                  labels={'customer_count': 'Number of Customers'},
                  color_discrete_sequence=colors)
    st.plotly_chart(fig4, use_container_width=True)
st.divider()


# --- Bar Chart: Spend Predictions by Job Industry Category ---
st.write("### Spend Predictions by Job Industry Category")
st.write("This bar chart shows the count of customers by job industry category for those predicted to spend or not spend in the next 90 days.")

# Group by job industry and spend_next_90_days_pred, then count
job_industry_counts = selected_df.groupby(['job_industry_category', 'spend_next_90_days_pred_C']).size().reset_index(name='customer_count')

fig5 = px.bar(job_industry_counts, 
              x='job_industry_category', 
              y='customer_count', 
              color='spend_next_90_days_pred_C', 
              barmode='group',
              title='Customer Spend Predictions by Job Industry Category',
              labels={'customer_count': 'Number of Customers'},
              color_discrete_sequence=colors)
st.plotly_chart(fig5, use_container_width=True)
st.divider()


# --- Line Area Chart: Spend Predictions by Property Valuation ---
st.write("### Spend Predictions by Property Valuation")
st.write("This chart shows the count of customers by property valuation for those predicted to spend or not spend in the next 90 days.")

property_valuation_counts = selected_df.groupby(['property_valuation', 'spend_next_90_days_pred']).size().reset_index(name='customer_count')

fig6 = px.area(property_valuation_counts, 
               x='property_valuation', 
               y='customer_count', 
               color='spend_next_90_days_pred', 
               title='Customer Spend Predictions by Property Valuation',
               labels={'customer_count': 'Number of Customers'},
               color_discrete_sequence=colors)
st.plotly_chart(fig6, use_container_width=True)
st.divider()


# --- Bar Chart: Spend Predictions by Age Group ---
# Create age groups based on age ranges
selected_df['age_group'] = pd.cut(selected_df['age'], 
                                  bins=[15, 30, 60, 100], 
                                  labels=['Young Adults (15-30)', 'Middle-aged (30-60)', 'Seniors (60+)'])

st.write("### Spend Predictions by Age Group")
st.write("This bar chart shows the count of customers by age group for those predicted to spend or not spend in the next 90 days.")

# Group by age group and spend_next_90_days_pred, then count
age_group_counts = selected_df.groupby(['age_group', 'spend_next_90_days_pred_C']).size().reset_index(name='customer_count')

fig7 = px.bar(age_group_counts, 
              x='age_group', 
              y='customer_count', 
              color='spend_next_90_days_pred_C', 
              barmode='group',
              title='Customer Spend Predictions by Age Group',
              labels={'customer_count': 'Number of Customers'},
              color_discrete_sequence=colors)
st.plotly_chart(fig7, use_container_width=True)
st.divider()

# Confusion matrix and ROC curve (if not 2018 Q1)
if selected != "2018 Q1 Prediction":
    st.write("### Confusion Matrix")
    st.write("The confusion matrix shows how well the model is performing in predicting customer spending (True Positive and True Negative rates). A higher diagonal value indicates better performance.")
    conf_matrix = confusion_matrix(selected_df['spend_next_90_days'], selected_df['spend_next_90_days_pred'])
    fig8 = ff.create_annotated_heatmap(conf_matrix, 
                                       x=['Predicted 0', 'Predicted 1'], 
                                       y=['Actual 0', 'Actual 1'], 
                                       colorscale=[[0.0, colors[1]], [1.0, colors[4]]], 
                                       showscale=True)
    st.plotly_chart(fig8)
    st.divider()

    # Calculate ROC curve
    st.write("### ROC Curve")
    st.write("The ROC curve shows the trade-off between sensitivity (true positive rate) and specificity (false positive rate). A model with a curve closer to the top left corner has better predictive power.")
    fpr, tpr, _ = roc_curve(selected_df['spend_next_90_days'], selected_df['spend_next_90_days_pred_proba'])
    # Create the ROC curve plot
    fig9 = go.Figure()
    fig9.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve', line=dict(color=colors[0], width=3)))
    fig9.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Model', line=dict(color=colors[4], dash='dash')))
    # Customize layout
    fig9.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        plot_bgcolor=colors[2],
        title_font=dict(size=20, color=colors[0]),
        xaxis_title_font=dict(size=16, color=colors[0]),
        yaxis_title_font=dict(size=16, color=colors[0])
    )
    st.plotly_chart(fig9)
    st.divider()
else:
    # User input for new customer prediction
    st.write("## Predict New Customer Behavior")
    st.write("Enter customer details below to predict the likelihood of them making a purchase within the next 90 days:")

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
    st.divider()

st.write("### Output Dataset")
with st.expander("Show Full Customers Prediction Dataset"):
    # Filters for spending prediction and state
    spend_filter = st.multiselect(
        "Filter by Spending Prediction",
        options=[1, 0],
        default=[1, 0],
        format_func=lambda x: 'Will Spend' if x == 1 else 'Will Not Spend')

    state_filter = st.multiselect(
        "Filter by State",
        options=selected_df['state'].unique(),
        default=selected_df['state'].unique())
    # Apply filters to the dataset
    filtered_df = selected_df[
        (selected_df['spend_next_90_days_pred'].isin(spend_filter)) &
        (selected_df['state'].isin(state_filter))]

    st.dataframe(filtered_df)
    st.write(f"##### Number of customers = **{len(filtered_df)}**")

st.divider()
st.markdown(f"""
    <style>
        .hover-div {{
            padding: 10px;
            border-radius: 10px;
            background-color: #2c413c;
            margin-bottom: 10px;
            display: flex;
            justify-content: center;  /* Centers horizontally */
            align-items: center;  /* Centers vertically */
            transition: background-color 0.3s ease, box-shadow 0.3s ease;
            cursor: pointer;  /* Makes the div look clickable */
            text-decoration: none;  /* Remove underline from the text */
        }}
        .hover-div:hover {{
            background-color: #1e7460; /* Slightly lighter background color on hover */
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.2); /* Adds a shadow on hover */
        }}
        h4 {{
            margin: 0; /* Remove any default margin */
            color: white; /* White text */
            text-align: center; /* Center the text */
        }}
    </style>
    <a href="https://github.com/MohamedHossam606/DEPI-Graduation-Project/tree/main/Notebooks/3_Prediction%20Notebooks" target="_blank" class="hover-div">
        <h4>View the full code Notebooks</h4>
    </a>""", unsafe_allow_html=True)
