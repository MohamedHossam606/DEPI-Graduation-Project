import streamlit as st
import pandas as pd
import requests
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Our Team", initial_sidebar_state='expanded')

col1, col2 = st.columns([3, 1])
with col1:
    st.title('🌟 Meet Our Team')
with col2:
    st.image("pages/assets/3_Meet_Our_Team/Team work-amico.png", use_column_width=True)
st.divider()

# LinkedIn icon URL
linkedin_icon = "https://img.icons8.com/?size=100&id=8808&format=png&color=FFFFFF"

st.markdown("<h3>Our Team Members</h3>", unsafe_allow_html=True)
team_members = [
    {"name": "Mohamed Hossam", "linkedin": "https://www.linkedin.com/in/mohamed-hossam-419b4727a/"},
    {"name": "Kareem Adel", "linkedin": "http://www.linkedin.com/in/kareem-adel-65441a1b0"},
    {"name": "Rawan Mohamed", "linkedin": "http://www.linkedin.com/in/rawan-anwar-181250322/"},
    {"name": "Rwaa Mohamed", "linkedin": "http://www.linkedin.com/in/rwaa-mohamed-a028372b1/"}
]

for member in team_members:
    st.markdown(f"""
        <style>
            .hover-div {{
                padding: 10px;
                border-radius: 10px;
                background-color: #2c413c;
                margin-bottom: 10px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                transition: background-color 0.3s ease, box-shadow 0.3s ease;
            }}
            .hover-div:hover {{
                background-color: #1e7460; /* Slightly lighter background color on hover */
                box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.2); /* Adds a shadow on hover */
            }}
            .linkedin-icon {{
                width: 35px; /* Bigger icon size */
                vertical-align: middle;
            }}
        </style>
        <div class="hover-div">
            <h4 style="margin-left: 15px; color: white;">{member['name']}</h4>
            <a href="{member['linkedin']}" target="_blank" style="margin-right: 25px;">
                <img src="{linkedin_icon}" class="linkedin-icon"/>
            </a>
        </div>
    """, unsafe_allow_html=True)

st.divider()

st.markdown("""
    <h3 style='color: #386641;'>We Value Your Feedback!</h3>
    <p style='font-size: 18px;'>Thank you for visiting our project page! We hope you enjoyed exploring our work. Your feedback is important to us, and we'd love to hear your thoughts, suggestions, or any questions you may have.</p>
""", unsafe_allow_html=True)

st.divider()

st.markdown("""
    <p style='font-size: 18px;'>A special thanks to <strong>Eng. Karim Ahmed</strong> for your valuable mentorship and supervision, which has been instrumental in our growth and success.</p>
""", unsafe_allow_html=True)
