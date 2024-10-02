import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

st.set_page_config(page_title="RFM Analysis", initial_sidebar_state='expanded')

col1, col2 = st.columns([3, 1])
with col1:
    st.title('RFM Analysis Dashboard')
with col2:
    st.image("pages/assets/1_RFM Analysis/Segmentation-cuate.png", use_column_width=True)

df = pd.read_csv("Notebooks/2_RFM Segment & LTV/RFM Segment & LTV.csv")

# Set up color palette
main_green = '#386641'
secondary_green = '#6A994E'
light_green = '#A7C957'
pastel_green = '#B7E4C7'
dark_olive_green = '#2F5233'
green_palette = [main_green, secondary_green, light_green, pastel_green, dark_olive_green]

tab1,tab2,tab3 = st.tabs(["RFM Data Overview", "Process", "LTV"])
with tab1:
    st.markdown('''
        <style>
        .header-title {
            font-size:30px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .sub-title {
            font-size:20px;
            font-weight: semi-bold;
            margin-bottom: 20px;
        }
        .description {
            font-size:16px;
            margin-bottom: 20px;
        }
        .icon-green {
            font-size: 18px;
        }
        </style>
    ''', unsafe_allow_html=True)

    st.markdown('<p class="header-title">📊 RFM Analysis Overview</p>', unsafe_allow_html=True)

    st.markdown('''<p class="sub-title">What is RFM?</p>''', unsafe_allow_html=True)
    st.markdown(
        '''
        <p class="description">
        RFM analysis is a technique used to evaluate and segment customers based on their transaction history. 
        This allows businesses to identify and focus on their most valuable customers. 
        </p>
        ''', unsafe_allow_html=True)

    st.markdown('<p class="sub-title">Components of RFM:</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('''
            <p class="description"><span class="icon-green">📅</span> <b>Recency (R):</b><br>
            How recently a customer made a purchase.<br>
            </p>
        ''', unsafe_allow_html=True)

    with col2:
        st.markdown('''
            <p class="description"><span class="icon-green">🔄</span> <b>Frequency (F):</b><br>
            How often a customer makes purchases.<br>
            </p>
        ''', unsafe_allow_html=True)

    with col3:
        st.markdown('''
            <p class="description"><span class="icon-green">💰</span> <b>Monetary Value (M):</b><br>
            Total amount spent by the customer.<br>
            </p>
        ''', unsafe_allow_html=True)

    st.divider()

    st.markdown('''     
    ##### **Benefits of RFM Analysis:**
    * **Customer Retention:** Helps identify which customers are at risk of leaving and need re-engagement.
    * **Personalized Marketing:** Allows for tailored messaging and offers based on different customer segments.
    * **Maximizing ROI:** Focuses resources on the most engaged and valuable customers.
    ''')

    st.markdown('''     
    ##### **Customer Segmentation:**
    Customers can be segmented into groups like Champions, Loyal Customers, or At-Risk based on scores. Only Recency and Frequency are emphasized for simplicity and to focus on customer engagement rather than monetary value.
    ''')

with tab2:
    st.markdown('''#### **Process of RFM Calculation:**''')
    st.markdown('''##### **Recency:** Calculate the difference in days between the most recent transaction and current date for each customer.''')
    code = '''
    current_date = pd.to_datetime("2017-12-31")

    recency_df = df.groupby('customer_id').agg({'transaction_date': lambda x: (current_date - x.max()).days}).reset_index()
    recency_df.columns = ['customer_id', 'recency']
    '''
    st.code(code, language="python")
    
    st.markdown('''##### **Frequency:**  Count the number of transactions for each customer.''')
    code = '''
    frequency_df = df.groupby('customer_id').agg({'transaction_id': 'count'}).reset_index()
    frequency_df.columns = ['customer_id', 'frequency']
    '''
    st.code(code, language="python")    
    
    st.markdown('''##### **Monetary:**  Sum the total spending for each customer.''')
    code = '''
    monetary_df = df.groupby('customer_id').agg({'list_price': 'sum'}).reset_index()
    monetary_df.columns = ['customer_id', 'monetary']
    '''
    st.code(code, language="python")
    
    st.markdown('''##### **Score Assignment:**  Customers are assigned scores (typically 1 to 5) for Recency, Frequency, and Monetary value. Higher scores indicate higher engagement or value.''')
    code = '''
    rfm_df['R_rank'] = pd.qcut(rfm_df['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm_df['F_rank'] = pd.qcut(rfm_df['frequency'], 5, labels=[1, 2, 3, 4, 5])
    rfm_df['M_rank'] = pd.qcut(rfm_df['monetary'], 5, labels=[1, 2, 3, 4, 5])
    '''
    st.code(code, language="python")    
    
    st.markdown('''##### **Customer Segmentation:** ''')
    code = '''
    seg_map = {
        r'[4-5][4-5]': 'Champions',            # High recency and frequency
        r'[3-4][3-5]': 'Loyal Customers',      # Moderately high recency and frequency
        r'[4-5][2-3]': 'Potential Loyalists',  # High recency but moderate frequency
        r'[1-2][3-5]': 'At Risk',              # Low recency but moderate to high frequency
        r'[1-2][1-2]': 'Hibernating',          # Low recency and frequency
        r'3[1-2]': 'New Customers',            # Moderate recency but low frequency (example)
        r'[4-5]1': 'Promising'                 # High recency but very low frequency (example)
    }

    rfm_df['Segment'] = rfm_df['R_rank'].astype(str) + rfm_df['F_rank'].astype(str)
    rfm_df['Segment'] = rfm_df['Segment'].replace(seg_map, regex=True)
    '''
    st.code(code, language="python")     
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
            text-align: center; /* Center the text */
        }}
    </style>
    <a href="https://github.com/MohamedHossam606/DEPI-Graduation-Project/blob/main/Notebooks/2_RFM%20Segment%20%26%20LTV/RFM%20Segment%20%26%20LTV.ipynb" target="_blank" class="hover-div">
        <h4 style="color: white;">View Full Code Notebook</h4>
    </a>""", unsafe_allow_html=True)
    st.divider()

with tab3:
    st.markdown('''
    <style>
    .sub-section-title {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .description {
        font-size: 16px;
        margin-bottom: 20px;
    }
    .highlight {
        font-weight: bold;
    }
    .icon-green {
        font-size: 18px;
    }
    </style>
    ''', unsafe_allow_html=True)

    # LTV and RFM explanation
    st.markdown('''##### LTV stands for Lifetime Value, a metric commonly used in business to estimate the total revenue a company can expect from a single customer over the course of their relationship with the company.''', unsafe_allow_html=True)

    st.markdown('''##### 📈 LTV Calculation:''')
    st.latex(r''' LTV = (Average \ Purchase \ Value) \times (Average \ Purchase \ Frequency) \times (Customer \ Tenure) ''')

    st.divider()
    st.markdown('''
    **Combining RFM and LTV for Targeted Marketing:**
    To improve efficiency, you can combine RFM and LTV analysis to refine your marketing strategy:''')

    st.markdown('''
    * 🟢 **High RFM + High LTV:** These are your best customers who spend a lot and are highly engaged. These customers should receive your best deals and exclusive offers. Focus on long-term retention.
                
    * 🟡 **High RFM + Low LTV:** These customers engage frequently but don't spend much. Target them with up-selling campaigns to increase their average purchase value, such as suggesting upgrades to their current bikes or premium accessories.
                
    * 🟠 **Low RFM + High LTV:** These are past high-value customers who have become inactive. Create win-back campaigns with personalized messages or special offers to bring them back. Highlight the value they used to gain from your store.
                
    * 🔴 **Low RFM + Low LTV:** These customers are less likely to engage or spend. Use low-cost marketing channels like email to send broad offers, and don't invest too heavily in retaining them.
    ''')
    st.divider()

images = [
    "pages/assets/1_RFM Analysis/Customer_Segmentation1.jpg",
    "pages/assets/1_RFM Analysis/Customer_Segmentation2.jpg",
]

# Custom CSS for button styles
st.markdown("""
    <style>
    .btn-style {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: background-color 0.3s ease-in-out;
    }
    .btn-style:hover {
        background-color: #45a049;
    }
    .slider-indicators {
        text-align: center;
        margin-top: 10px;
    }
    .slider-indicators span {
        height: 15px;
        width: 15px;
        margin: 0 5px;
        display: inline-block;
        background-color: #bbb;
        border-radius: 50%;
    }
    .slider-indicators .active {
        background-color: #717171;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for image index and auto-slide timer
if "carousel_index" not in st.session_state:
    st.session_state.carousel_index = 0
# Display the current image
st.image(images[st.session_state.carousel_index], width=700)
# Navigation buttons (Previous/Next)
prev, _, next = st.columns([1, 10, 1])
# Handle the previous button click
if prev.button("◀", key="prev", help="Previous image", type="primary"):
    st.session_state.carousel_index = (st.session_state.carousel_index - 1) % len(images)
# Handle the next button click
if next.button("▶", key="next", help="Next image", type="primary"):
    st.session_state.carousel_index = (st.session_state.carousel_index + 1) % len(images)
# Extra description or metadata
st.write(f"Image {st.session_state.carousel_index + 1} of {len(images)}")

with st.expander("Show Full Customers Segmentation Dataset"):
    st.write("The table below shows the key RFM metrics (Recency, Frequency, and Monetary Value) for each customer. RFM segmentation is a powerful technique for understanding customer behavior.")

    segment_filter = st.multiselect(
        "Filter by Segment",
        options=df['Segment'].unique(),
        default=df['Segment'].unique())
    
    state_filter = st.multiselect(
        "Filter by State",
        options=df['state'].unique(),
        default=df['state'].unique())
    
    # Apply filters to the dataset
    filtered_df = df[
        (df['Segment'].isin(segment_filter)) &
        (df['state'].isin(state_filter))]

    st.dataframe(filtered_df)
    st.write(f"##### Number of customers = **{len(filtered_df)}**")

st.divider()

### Graph 1: Distribution of Customers by RFM Segment
st.subheader("Customer Distribution by RFM Segment")
st.write("This bar chart shows the distribution of customers across different RFM segments. It helps identify the concentration of customers in each segment, helping tailor marketing strategies for each group.")
segment_counts = df['Segment'].value_counts().reset_index()
segment_counts.columns = ['Segment', 'count']

fig1 = px.bar(segment_counts, 
              x='Segment', y='count', 
              labels={'Segment': 'RFM Segment', 'count': 'Customer Count'},
              title='Customer Distribution by RFM Segment',
              color_discrete_sequence=green_palette)
st.plotly_chart(fig1)
st.divider()

### Graph 2: Proportion of RFM Segments (Pie Chart)
col1, col2 = st.columns(2)

with col1:
    st.subheader("Proportion of RFM Segments")
    st.write("The pie chart provides a quick visual of how customers are distributed across different RFM segments, offering insight into segment size and the potential need for specific attention.")
    fig2 = px.pie(df, names='Segment', title='Proportion of Customers by RFM Segment',
                  color_discrete_sequence=green_palette, hole=0.3)
    st.plotly_chart(fig2)

with col2:
    st.subheader("Customer Segmentation Overview (Treemap)")
    st.write("The treemap gives a hierarchical view of customer segments, highlighting the relative proportion of each segment more clearly.")
    fig3 = px.treemap(segment_counts, path=['Segment'], values='count', 
                      title='Customer Segmentation Overview', 
                      color='count',
                      color_continuous_scale=green_palette)
    st.plotly_chart(fig3)
st.divider()

### Graph 4: Customer Count by Wealth Segment and RFM Segment
st.subheader("Customer Count by Wealth Segment and RFM Segment")
st.write("This histogram compares customer counts across RFM segments, with an added layer of wealth segment information. It provides insight into how wealth levels intersect with RFM segmentation.")
fig4 = px.histogram(df, x='Segment', color='wealth_segment', 
                    barmode='group', title='Customer Count by Wealth Segment and RFM Segment',
                    color_discrete_sequence=green_palette)
st.plotly_chart(fig4)
st.divider()

### Graph 5: Create a bar chart
st.subheader("Customer Segments by State")
st.write("This bar chart visualizes the distribution of customer segments across different states. Each segment is represented by a distinct color, allowing for easy identification of how each segment is represented geographically.")
segment_state_counts = df.groupby(['Segment', 'state']).size().reset_index(name='count')
fig5 = px.bar(segment_state_counts, x='state', y='count', color='Segment', 
                            title='Customer Segments by State',
                            labels={'state': 'State', 'count': 'Customer Count'},
                            color_discrete_sequence=green_palette)
# Update layout for better appearance
fig5.update_layout(barmode='stack')  # Use 'group' for grouped bars
st.plotly_chart(fig5)
st.divider()

### Graph 6: Recency Distribution
st.subheader("Recency Distribution")
st.write("The histogram shows the distribution of recency scores (days since last purchase). It helps understand how frequently customers are making purchases, where lower recency values indicate more recent engagement.")
fig6 = px.histogram(df, x='recency', nbins=20, title='Recency Distribution',
                    color_discrete_sequence=[main_green], marginal='box')
st.plotly_chart(fig6)
st.divider()

### Graph 7: Frequency Distribution
st.subheader("Frequency Distribution")
st.write("This chart highlights how often customers are making purchases. A higher frequency score indicates a more loyal or engaged customer.")
fig7 = px.histogram(df, x='frequency', nbins=20, title='Frequency Distribution',
                    color_discrete_sequence=[light_green], marginal='box')
st.plotly_chart(fig7)
st.divider()

### Graph 8: Monetary Value Distribution
st.subheader("Monetary Value Distribution")
st.write("This histogram shows the distribution of total spendings by customers. Higher values indicate that the customer has spent more money overall, which is crucial for identifying high-value customers.")
fig8 = px.histogram(df, x='monetary', nbins=20, title='Monetary Value Distribution',
                    color_discrete_sequence=[pastel_green], marginal='box')
st.plotly_chart(fig8)
st.divider()

# ### Graph 9: Boxplot of Monetary Values Across Segments
# st.subheader("Monetary Value by Segment")
# st.write("The boxplot shows how the monetary value varies across different RFM segments, providing insights into the spending behavior of customers in each segment.")
# fig9 = px.box(df, x='Segment', y='monetary', title='Monetary Value by Segment',
#               color='Segment', color_discrete_sequence=green_palette)
# st.plotly_chart(fig9)
# st.divider()

### Graph 10: LTV Distribution
st.subheader("Lifetime Value (LTV) Distribution")
st.write("This chart shows the distribution of Lifetime Value (LTV), which estimates the total worth of a customer to the company over the entire relationship.")
fig10 = px.histogram(df, x='LTV', nbins=20, title='Lifetime Value Distribution',
                    color_discrete_sequence=[secondary_green], marginal='box')
st.plotly_chart(fig10)
st.divider()

### Graph 11: Average LTV by Segment
st.subheader("Average Lifetime Value by Segment")
st.write("This bar chart shows the average lifetime value (LTV) of customers in each segment. It helps identify which segments generate the most long-term value.")
df_avg_ltv = df.groupby('Segment')['LTV'].mean().reset_index()
fig11 = px.bar(df_avg_ltv, x='Segment', y='LTV', title='Average Lifetime Value by Segment',
               color_discrete_sequence=green_palette)
st.plotly_chart(fig11)
st.divider()

### Graph 12: Correlation Heatmap Between RFM Features
st.subheader("Correlation Between RFM Features")
st.write("The heatmap below shows the correlation between the key RFM features: Recency, Frequency, and Monetary value. It helps identify relationships between these features.")
corr = df[['recency', 'frequency', 'monetary']].corr()
fig14 = ff.create_annotated_heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index), 
                                    colorscale='Greens', showscale=True, hoverinfo="z")
st.plotly_chart(fig14)
