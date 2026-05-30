import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="EduPro Forecast Dashboard",
    page_icon="📚",
    layout="wide"
)

# =========================
# LOAD MODELS
# =========================

demand_model = joblib.load("demand_model.pkl")
revenue_model = joblib.load("revenue_model.pkl")

# =========================
# TITLE
# =========================

st.title("📚 EduPro Course Demand & Revenue Forecasting")

st.markdown("""
This dashboard predicts:

- Course Enrollment Demand
- Course Revenue Forecast

using Machine Learning models trained on EduPro data.
""")

# =========================
# INPUT SECTION
# =========================

st.header("🎯 Enter Course Details")

col1, col2 = st.columns(2)

with col1:
    price = st.number_input(
        "Course Price",
        min_value=0.0,
        value=100.0
    )

    duration = st.number_input(
        "Course Duration (Hours)",
        min_value=1,
        value=10
    )

    rating = st.slider(
        "Course Rating",
        0.0,
        5.0,
        4.0
    )

with col2:
    teacher_rating = st.slider(
        "Teacher Rating",
        0.0,
        5.0,
        4.0
    )

    experience = st.number_input(
        "Years Of Experience",
        min_value=0,
        value=5
    )

# =========================
# PREDICTION
# =========================

if st.button("Predict"):

    input_data = pd.DataFrame(
        [[
            price,
            duration,
            rating,
            0,
            0,
            0,
            teacher_rating,
            experience
        ]],
        columns=[
            "CoursePrice",
            "CourseDuration",
            "CourseRating",
            "CourseCategory",
            "CourseType",
            "CourseLevel",
            "TeacherRating",
            "YearsOfExperience"
        ]
    )

    demand_prediction = demand_model.predict(input_data)[0]
    revenue_prediction = revenue_model.predict(input_data)[0]

    st.success(
        f"📈 Predicted Enrollment Count: {round(demand_prediction)}"
    )

    st.success(
        f"💰 Predicted Revenue: ${round(revenue_prediction, 2)}"
    )

# =========================
# FEATURE IMPORTANCE
# =========================

st.header("📊 Feature Importance Analysis")

try:
    importance_df = pd.read_csv("feature_importance.csv")

    st.bar_chart(
        importance_df.set_index("Feature")
    )

except:
    st.warning(
        "feature_importance.csv not found."
    )

# =========================
# CATEGORY COMPARISON
# =========================

st.header("📈 Category-Level Demand Comparison")

try:
    category_data = pd.read_csv(
        "category_comparison.csv"
    )

    st.subheader(
        "Enrollment Comparison by Category"
    )

    st.bar_chart(
        category_data.set_index(
            "CourseCategory"
        )["EnrollmentCount"]
    )

    st.subheader(
        "Revenue Comparison by Category"
    )

    st.bar_chart(
        category_data.set_index(
            "CourseCategory"
        )["CourseRevenue"]
    )

except:
    st.warning(
        "category_comparison.csv not found."
    )

# =========================

# =========================


category_data = pd.read_csv("category_comparison.csv")

category_map = {
    0: "Data Science",
    1: "Programming",
    2: "Business",
    3: "Marketing",
    4: "Design"
}

category_data["CourseCategory"] = (
    category_data["CourseCategory"]
    .map(category_map)
)