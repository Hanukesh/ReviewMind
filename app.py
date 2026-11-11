import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

from datetime import datetime

# Database imports
from database import (
    create_table,
    add_review,
    get_reviews,
    clear_reviews
)


# PAGE CONFIGURATION
st.set_page_config(
    page_title="ReviewMind",
    page_icon="🤖",
    layout="wide"
)


# LOAD ML MODEL
if not os.path.exists("sentiment_model.pkl"):

    st.error("Model not found! Run improve_model.py")
    st.stop()

model = joblib.load("sentiment_model.pkl")


# DATABASE INITIALIZATION
create_table()


# CUSTOM CSS
st.markdown("""
<style>

/* MAIN APPLICATION */
.stApp {
    background-color: #0E1117;
    color: white;
}

.main-title {
    font-size: 40px;
    font-weight: bold;
    color: #8B9CFB;
}

/* SIDEBAR BACKGROUND */
[data-testid="stSidebar"] {
    background: #171B2B;
    border-right: 1px solid #303850;
}

/* SIDEBAR TITLE */
[data-testid="stSidebar"] h1 {
    font-size: 25px;
    font-weight: 800;
    color: #A5B4FC;
    letter-spacing: 0.5px;
}

/* SIDEBAR TEXT */
[data-testid="stSidebar"] p {
    font-size: 14px;
    font-weight: 500;
    color: #B8C0D9;
}

/* NAVIGATION HEADING */
[data-testid="stSidebar"] h2 {
    font-size: 17px;
    font-weight: 700;
    color: #FFFFFF;
}

/* NAVIGATION SPACING */
[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 10px;
}

/* NAVIGATION OPTIONS */
[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: #22283A;
    padding: 10px 14px;
    border-radius: 9px;
    border: 1px solid #303850;
    transition: all 0.2s ease;
    cursor: pointer;
}

/* HOVER EFFECT */
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: #343D5C;
    border-color: #818CF8;
}

/* NAVIGATION FONT */
[data-testid="stSidebar"] div[role="radiogroup"] p {
    color: #E2E8F0;
    font-size: 14px;
    font-weight: 600;
}

/* RADIO BUTTON */
[data-testid="stSidebar"] input[type="radio"] {
    accent-color: #818CF8;
}

</style>
""", unsafe_allow_html=True)


# SIDEBAR NAVIGATION
with st.sidebar:

    st.title("🤖 ReviewMind")

    st.caption("AI Review Analyzer")

    st.divider()

    st.subheader("Navigation")

    page = st.radio(
        "Navigation",
        ["Home", "Sentiment Analysis", "Dashboard", "About"],
        label_visibility="collapsed"
    )

    st.divider()

    st.caption("Python | NLP | Machine Learning")


# HOME PAGE
if page == "Home":

    st.markdown(
        '<div class="main-title">ReviewMind</div>',
        unsafe_allow_html=True
    )

    st.subheader("AI-Powered Customer Review Analyzer")

    st.write(
        "Analyze customer feedback using "
        "Artificial Intelligence and NLP."
    )

    st.info(
        "Navigate to Sentiment Analysis to analyse customer reviews."
    )


# SENTIMENT ANALYSIS PAGE
elif page == "Sentiment Analysis":

    st.title("🔍 Sentiment Analysis")

    review = st.text_area(
        "Enter your customer review",
        height=150
    )

    if st.button("Analyze Review"):

        if review.strip():

            # AI prediction
            prediction = model.predict([review])[0]

            # Confidence calculation
            probabilities = model.predict_proba([review])[0]

            confidence = max(probabilities) * 100

            # SAVE REVIEW IN DATABASE
            add_review(
                review,
                prediction,
                round(confidence, 2),
                datetime.now().strftime(
                    "%d-%m-%Y %H:%M:%S"
                )
            )

            st.subheader("Analysis Result")

            if prediction == "positive":

                st.success("😊 Positive Review")

            elif prediction == "negative":

                st.error("😡 Negative Review")

            else:

                st.info("😐 Neutral Review")

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            st.success("Review saved successfully!")

        else:

            st.warning("Please enter a review!")


# DASHBOARD PAGE
elif page == "Dashboard":

    st.title("📊 ReviewMind Analytics Dashboard")

    # GET DATA FROM DATABASE
    df = get_reviews()

    if df.empty:

        st.info(
            "No reviews available. Analyse some reviews first!"
        )

    else:

        # Calculate statistics
        total = len(df)

        positive = (df["sentiment"] == "positive").sum()

        negative = (df["sentiment"] == "negative").sum()

        neutral = (df["sentiment"] == "neutral").sum()

        # METRIC CARDS
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Reviews", total)

        col2.metric("Positive 😊", positive)

        col3.metric("Negative 😡", negative)

        col4.metric("Neutral 😐", neutral)

        st.divider()

        # PIE CHART
        st.subheader("Sentiment Distribution")

        counts = df["sentiment"].value_counts()

        fig, ax = plt.subplots()

        ax.pie(
            counts,
            labels=counts.index,
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title("Customer Sentiment Distribution")

        st.pyplot(fig)

        plt.close(fig)

        # BAR CHART
        st.subheader("Sentiment Comparison")

        chart_data = pd.DataFrame({
            "Sentiment": ["Positive", "Negative", "Neutral"],
            "Count": [positive, negative, neutral]
        })

        st.bar_chart(
            chart_data.set_index("Sentiment")
        )

        st.divider()

        # REVIEW HISTORY
        st.subheader("📋 Customer Review History")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        # CLEAR HISTORY
        if st.button("🗑️ Clear Review History"):

            clear_reviews()

            st.success("Review history cleared successfully!")

            st.rerun()


# ABOUT PAGE
elif page == "About":

    st.title("ℹ️ About ReviewMind")

    st.write("""
    ReviewMind is an AI-powered customer review
    sentiment analysis application.

    It uses Natural Language Processing and
    Machine Learning to classify customer
    reviews into three categories.
    """)

    st.subheader("Technologies Used")

    st.write("""
    - Python
    - Natural Language Processing
    - TF-IDF
    - Logistic Regression
    - Scikit-learn
    - Streamlit
    - Pandas
    - Matplotlib
    - SQLite Database
    """)

    st.success("Developed as a B.E. AIML project.")