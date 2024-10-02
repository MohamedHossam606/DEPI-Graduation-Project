import streamlit as st
import pandas as pd
from streamlit_lottie import st_lottie

All_Customers_df = pd.read_csv("Notebooks/1_Preprocessing Notebooks/Final Sheets/customer_details.csv")
products_df = pd.read_csv("Notebooks/1_Preprocessing Notebooks/Final Sheets/products_details.csv")
Transactions_df = pd.read_csv("Notebooks/1_Preprocessing Notebooks/Final Sheets/transactions_details.csv")

st.set_page_config(page_title="Home Page", initial_sidebar_state='expanded')

st.image("pages/assets/Main/DEPI-Graduation-Project.jpg")

tab1,tab2 = st.tabs(["About Dataset", "Data Preparation"])
with tab1:
    st.markdown('''
    ##### The dataset provides information on bike store transactions for the year 2017 in Australia. The dataset contains the followin tables
                
    #### **Customer Details Table:**
    * It contains details about the customers' age, gender, Job, and other demographic information, and also provides geographical information for each customer.
    * **Meta Data**:
    ''')
    Customers_Sheets = pd.DataFrame({
    'Column':['customer_id', 'first_name', 'last_name', 'gender', 'DOB', 'past_3_years_bike_related_purchases', 'job_title', 'job_industry_category', 'wealth_segment', 'deceased_indicator','owns_car', 'tenure', 'customer_id', 'address', 'postcode', 'state', 'country', 'property_valuation'],
    'Datatype':['Number', 'Text', 'Text', 'Text', 'Date', 'Number', 'Text', 'Text', 'Text', 'Text', 'Text', 'Number', 'Number', 'Text', 'Text', 'Text', 'Text', 'Number'],
    'Description':['Unique identifier for customers (Primary Key)',
                    'Customer\'s first name',
                    'Customer\'s last name',
                    'Customer\'s gender',
                    'Customer\'s date of birth',
                    'Number of bike-related purchases in the last 3 years',
                    'Customer\'s job title',
                    'The industry category in which the customer works',
                    'Classification based on customer\'s wealth (Mass, Affluent, High Net Worth)',
                    'Indicates if the customer is deceased (Y for yes, N for no)',
                    'Indicates if the customer owns a car (yes or No)',
                    'The length of time (in years) the customer has been associated with store.',
                    'Unique identifier for customers (Foreign Key).',
                    'The full address of the customer (street number and name).',
                    'The postal code associated with the customer\'s address.',
                    'The state where the customer resides (New South Wales, QLD, VIC).',
                    'The country of residence (Australia in this case).',
                    'A numeric value representing the property valuation rating (possibly on a scale of 1-12).'
                    ]})
    st.dataframe(Customers_Sheets)

    with st.expander("Show Customers Details Sample Data"):
        st.dataframe(All_Customers_df.sample(10),use_container_width=True)
        st.write(f"###### Customers Dataframe Shape = **{All_Customers_df.shape}**")

    st.markdown('''     
    #### **Transactions Table:**
    * It contains transaction details such as transaction ID, date, product ID, and total purchase amount, providing insights into customer purchasing behavior and store performance.

    * **Meta Data**:
    ''')
    Transactions_Sheet = pd.DataFrame({
        'Column':['transaction_id', 'unique_product_id', 'customer_id', 'transaction_date', 'online_order', 'order_status', 'list_price', 'standard_cost'],
        'Datatype':['Number', 'Number', 'Number', 'Short date', 'Text', 'Text', 'Currency', 'Currency'],
        'Description':['Unique identifier for each transaction. (Primary key)',
                        'Identifies the product involved in the transaction.(Foreign Key)',
                        'Identifies the customer involved in the transaction.(Foreign Key)',
                        'The date when the transaction occurred.',
                        'Indicates whether the transaction was an online order (TRUE for online, FALSE for offline).',
                        'The status of the order (Approved, Cancelled).',
                        'The product’s price at the time of the transaction.',
                        'The cost incurred by the company to produce or purchase the product.']})
    
    st.dataframe(Transactions_Sheet)
    
    with st.expander("Show Transactions Sample Data"):
        st.dataframe(Transactions_df.sample(10),use_container_width=True)
        st.write(f"###### Transactions Dataframe Shape = **{Transactions_df.shape}**")

    st.markdown('''     
    #### **Products Table:**
    * It contains stores detailed information about each product .

    * **Meta Data**:
    ''')
    Products_Sheet = pd.DataFrame({
        'Column': ['unique_product_id', 'product_id', 'brand', 'product_line', 'product_class', 'product_size', 'model'],
        'Datatype': ['Number', 'Number', 'Text', 'Text', 'Text', 'Text', 'Number'],
        'Description': [
            'Unique identifier for each product',
            'Identifier shared by variations of a product (e.g., different brands/sizes)',
            'Brand name of the product',
            'Specifies the product line, such as Road, Touring, Standard, or Mountain.',
            'Classification of the product in terms of quality or level, such as high, medium, or low.',
            'The size of the product, for example, large, medium, or small.',
            'Model name or number']})
    st.dataframe(Products_Sheet)

    with st.expander("Show Products Sample Data"):
        st.dataframe(products_df.sample(10),use_container_width=True)
        st.write(f"###### Customers Address Dataframe Shape = **{products_df.shape}**")

with tab2:
    st.markdown('''
    #### **Data Preparation:**
                
    In the beginning, the dataset contained four sheets with valuable insights into customer demographics, addresses, and transactions.
    * **Customer Demographic:** Contains details about the customers' age, gender, Job, and other demographic information.
    * **Customer Address:** Provides the geographical information for each customer, including address, state, and postal code, which can be useful for regional analysis and customer distribution.
    * **New Customer List:** Newly added customers. It contains both customer demographic data and addresses.
    * **Transactions:** Contains transaction details such as transaction ID, date, product details, and total purchase amount, providing insights into customer purchasing behavior.
                   
    #### **Data Merging**
    * **First:** we merged the Customer Demographic and Customer Address to consolidate all customer details into one table
    * **Second:** There were two customers separate tables:
        * Old Customers Table: Included customer records with IDs.
        * New Customers Table: Contained new customer records without IDs.
            * Solution: we implemented the following steps:
                1. Assign IDs to New Customers
                2. Append New Customers to Old Customers Table
    ''')
    st.divider()
    st.markdown("#### **Data Quality (Data Preparation)**")
    st.image("pages/assets/Main/Data Preparation process.png")
    
    st.markdown('''
    #### **Customer Details Table**
    ##### **Problem 1:** Invalid Data
    * **Invalid Tenure for New Customers:** The tenure column had unrealistic values, like 15 or 9 years, for new customers.
        * **Solution:** we assumed that they are truly new customers and that the issue lies with the tenure column being incorrect. we corrected these values by filling the tenure column with 0 for all new customers.

    * **Unrealistic Age Values:** Some age values, such as 174, were clearly invalid and impacted data quality.
        * **Solution:** I deleted the rows containing values to maintain the accuracy of the dataset.

    * **State Abbreviation:** The 'state' column contained both full state names and abbreviations, like 'NSW' and 'New South Wales’.
        * **Solution:** I replaced all abbreviations with full state names for consistency.
            
    * **Gender:** The 'gender’ column had inconsistent values such as 'F', 'Femal', 'M', and ‘U’.
        * **Solution:** Standardized all values to 'Female', 'Male', and 'Unknown' to ensure consistency across the dataset.

    ##### **Problem 2:** Missing Values
    * **Missing Last Names:** Some customer records had missing values in the last_name column.
        * **Solution:** I created a new name column by concatenating the first_name and last_name columns.
                
    * **Missing Job Titles and Industry Categories:** Some records had missing values, creating data gaps.
        * **Solution:** I filled missing values in the job_title and job_industry_category columns with 'N/A' to indicate that this information was unavailable.

    * **Missing Age values:** The age column had missing values in some customer records.
        * **Solution:** Applied the K-Nearest Neighbors (KNN) imputer method to fill in the missing age values. This method considers the 3 nearest neighbors based on the existing data and imputes the missing values with approximate values derived from similar customers.          
    ''')
    
    st.divider()

    st.markdown('''
    #### **Transactions Table**
    ##### **Problem 1:** Duplicate Product IDs
    * **Multiple Rows with Same Product ID:** Found instances where the Product ID was duplicated across several rows, but the details in columns like brand, Product line, Product class, and Product size were different.
    ''')
    st.image("pages/assets/Main/Duplicate-Product-IDs.png")
    
    st.markdown('''
    * **Solution:**
                1. **Creating Numeric Codes for Product Attributes:** To streamline the process of generating the Unique Product ID, I assigned numeric values to each categorical attribute. This helped simplify and standardize the encoding of product features.
    ''')
    st.image("pages/assets/Main/Mapping-Process.png")
    
    st.markdown('''
                2. **Created a Unique Product ID column:**
                The identifier is composed of 11 digits:
                   * Digits 1-3: Represent the original Product ID.
                   * Digit 4: Represents the brand.
                   * Digit 5: Represents the Product line.
                   * Digit 6: Represents the Product class.
                   * Digit 7: Represents the Product size.
                   * Digits 8-11: Represent the Model year (the year of the product release).
    ''') 
    left_co, cent_co, last_co = st.columns(3)
    with cent_co:
        st.image("pages/assets/Main/product ID.png", width=250)
        
    st.markdown('''
    ##### **Problem 2:** Missing Values
    Several columns, including brand, product_line, product_class, and product_size, had missing values, leading to incomplete records in the dataset.
    1. **Fill missing categorical values with 'N/A’:**
    Used .fillna('N/A') to ensure that all missing categorical values are replaced with 'N/A' for consistency and better handling in analysis.

    2. **Fill missing numerical values in key columns using KNN Imputer:**
    Applied the KNN imputation method to fill the missing values in online_order, standard_cost, and product_first_sold_date. By leveraging the 3 nearest neighbors, missing values were imputed with the most approximate values based on the patterns in the data.

    ##### **Problem 2:** Data Separation
    * **Problem:** The transactions table contained product details, which made it difficult to manage product information effectively.
        * **Solution:** To resolve this issue, the following steps were taken:
            1. **Split Product Details from Transactions:** Extracted columns related to products, such as product ID, brand, product line, etc., into products table.
            
            2. **Establish Foreign Key Relationships:** Linked the new product table with the transactions table using the product ID as a foreign key        
    ''')
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
    <a href="https://github.com/MohamedHossam606/DEPI-Graduation-Project/tree/main/Notebooks/1_Preprocessing%20Notebooks" target="_blank" class="hover-div">
        <h4 style="color: white;">View Full Data Preparation Notebooks</h4>
    </a>""", unsafe_allow_html=True)
st.divider()

st.write('## **ERD Diagram:**')
st.image("pages/assets/Main/Bikes-Store-ERD.png")
st.divider()

st.write('## **Dashboard**')

images = [
    "pages/assets/Main/Dashboard/Bickes Store Visualization_page-0002.jpg",
    "pages/assets/Main/Dashboard/Bickes Store Visualization_page-0003.jpg",
    "pages/assets/Main/Dashboard/Bickes Store Visualization_page-0004.jpg",
    "pages/assets/Main/Dashboard/Bickes Store Visualization_page-0005.jpg"
]

# Custom CSS for modern button styles
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
    <a href="https://app.powerbi.com/links/sgzZ23FXw_?ctid=878ae732-59c5-40e3-8d49-91e7988bccfd&pbi_source=linkShare" target="_blank" class="hover-div">
        <h4 style="color: white;">View Dashboard</h4>
    </a>""", unsafe_allow_html=True)

st.divider()

st.write('## **Presentation**')
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
    <a href="https://drive.google.com/drive/folders/1TL_3geXkXLWuHkv0V8MHhyG0jTeUY3c_?usp=sharing" target="_blank" class="hover-div">
        <h4 style="color: white;">View Our Presentation</h4>
    </a>""", unsafe_allow_html=True)
