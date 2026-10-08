import streamlit as st
import pandas as pd
import joblib

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Road Accident Severity Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("accident_severity_random_forest.pkl")

model = load_model()

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 19px;
    color: #666;
    margin-bottom: 25px;
}

.section-title {
    font-size: 27px;
    font-weight: 650;
    margin-top: 20px;
}

.prediction-card {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin: 15px 0;
}

.prediction-text {
    font-size: 32px;
    font-weight: 700;
}

.metric-card {
    padding: 18px;
    border-radius: 12px;
    background-color: #f5f7fa;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("📌 Project Information")

    st.markdown("""
    **Project:**  
    Road Accident Severity Prediction

    **Algorithm:**  
    Random Forest Classifier

    **Dataset:**  
    Indian Road Accident Dataset
    """)

    st.divider()

    st.subheader("🎯 Prediction Classes")

    st.markdown("""
    🟢 **Minor**

    🟠 **Major**

    🔴 **Fatal**
    """)

    st.divider()

    st.subheader("📊 Model Performance")

    st.metric("Accuracy", "52.05%")
    st.metric("Weighted F1", "42.17%")
    st.metric("ROC-AUC", "0.4892")

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🚗 Road Accident Severity Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict accident severity using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# =========================================================
# INPUT SECTION
# =========================================================

st.markdown(
    '<div class="section-title">📝 Accident Details</div>',
    unsafe_allow_html=True
)

st.write("Enter the accident conditions below and click **Predict Severity**.")

col1, col2, col3 = st.columns(3)

# ---------------------------------------------------------
# COLUMN 1
# ---------------------------------------------------------

with col1:

    city = st.text_input(
        "City",
        value="Hyderabad"
    )

    state = st.text_input(
        "State",
        value="Telangana"
    )

    latitude = st.number_input(
        "Latitude",
        value=17.3850
    )

    longitude = st.number_input(
        "Longitude",
        value=78.4867
    )

    hour = st.slider(
        "Hour of Accident",
        0,
        23,
        18
    )

    day_of_week = st.selectbox(
        "Day of Week",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )

    is_weekend = st.selectbox(
        "Weekend?",
        [0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )

# ---------------------------------------------------------
# COLUMN 2
# ---------------------------------------------------------

with col2:

    road_type = st.selectbox(
        "Road Type",
        [
            "urban",
            "rural",
            "highway"
        ]
    )

    lanes = st.number_input(
        "Number of Lanes",
        1,
        10,
        3
    )

    traffic_signal = st.selectbox(
        "Traffic Signal",
        [0, 1],
        format_func=lambda x:
            "Present" if x == 1 else "Not Present"
    )

    weather = st.selectbox(
        "Weather",
        [
            "clear",
            "rain",
            "fog",
            "storm"
        ]
    )

    visibility = st.selectbox(
        "Visibility",
        [
            "high",
            "medium",
            "low"
        ]
    )

    temperature = st.number_input(
        "Temperature (°C)",
        value=28.0
    )

    traffic_density = st.selectbox(
        "Traffic Density",
        [
            "low",
            "medium",
            "high"
        ]
    )

# ---------------------------------------------------------
# COLUMN 3
# ---------------------------------------------------------

with col3:

    cause = st.selectbox(
        "Main Cause",
        [
            "overspeeding",
            "drunk driving",
            "distraction",
            "poor road",
            "weather"
        ]
    )

    vehicles_involved = st.number_input(
        "Vehicles Involved",
        1,
        20,
        2
    )

    is_peak_hour = st.selectbox(
        "Peak Hour?",
        [0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )

    year = st.number_input(
        "Year",
        2020,
        2030,
        2025
    )

    month = st.selectbox(
        "Month",
        list(range(1, 13)),
        index=9
    )

# =========================================================
# INPUT DATA
# =========================================================

input_data = pd.DataFrame([{

    "city": city,
    "state": state,
    "latitude": latitude,
    "longitude": longitude,
    "hour": hour,
    "day_of_week": day_of_week,
    "is_weekend": is_weekend,
    "road_type": road_type,
    "lanes": lanes,
    "traffic_signal": traffic_signal,
    "weather": weather,
    "visibility": visibility,
    "temperature": temperature,
    "traffic_density": traffic_density,
    "cause": cause,
    "vehicles_involved": vehicles_involved,
    "is_peak_hour": is_peak_hour,
    "year": year,
    "month": month

}])

# =========================================================
# PREDICT BUTTON
# =========================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Accident Severity",
    use_container_width=True,
    type="primary"
)

if predict_button:

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability
    probabilities = model.predict_proba(input_data)[0]

    classes = model.classes_

    probability_df = pd.DataFrame({
        "Severity": classes,
        "Probability": probabilities
    })

    probability_df["Probability (%)"] = (
        probability_df["Probability"] * 100
    ).round(2)

    # =====================================================
    # RESULT
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-title">🎯 Prediction Result</div>',
        unsafe_allow_html=True
    )

    if prediction == "fatal":

        st.error(
            "🔴 PREDICTED ACCIDENT SEVERITY: FATAL"
        )

    elif prediction == "major":

        st.warning(
            "🟠 PREDICTED ACCIDENT SEVERITY: MAJOR"
        )

    else:

        st.success(
            "🟢 PREDICTED ACCIDENT SEVERITY: MINOR"
        )

    # =====================================================
    # PROBABILITY
    # =====================================================

    st.markdown(
        '<div class="section-title">📊 Prediction Probability</div>',
        unsafe_allow_html=True
    )

    chart_data = probability_df.set_index(
        "Severity"
    )["Probability (%)"]

    st.bar_chart(chart_data)

    st.dataframe(
        probability_df[
            ["Severity", "Probability (%)"]
        ],
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # ACCIDENT SUMMARY
    # =====================================================

    st.markdown(
        '<div class="section-title">📋 Accident Information</div>',
        unsafe_allow_html=True
    )

    summary = pd.DataFrame({
        "Parameter": [
            "City",
            "State",
            "Road Type",
            "Weather",
            "Visibility",
            "Traffic Density",
            "Main Cause",
            "Vehicles Involved",
            "Peak Hour"
        ],

        "Value": [
            city,
            state,
            road_type,
            weather,
            visibility,
            traffic_density,
            cause,
            vehicles_involved,
            "Yes" if is_peak_hour else "No"
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

# =========================================================
# ABOUT MODEL
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">🤖 About the Model</div>',
    unsafe_allow_html=True
)

st.write("""
The system uses a **Random Forest Classifier** to predict
road accident severity based on accident, road, weather,
traffic and environmental characteristics.
""")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        "Algorithm",
        "Random Forest"
    )

with m2:
    st.metric(
        "Training Samples",
        "16,000"
    )

with m3:
    st.metric(
        "Test Samples",
        "4,000"
    )

# =========================================================
# LIMITATIONS
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">⚠️ Limitations</div>',
    unsafe_allow_html=True
)

st.info("""
The experimental results indicate that the available dataset
contains limited predictive information for distinguishing
accident severity classes. Therefore, this application is
intended as a machine-learning prototype and should not be
used as a substitute for professional road-safety assessment.
""")

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Road Accident Severity Prediction | "
    "Random Forest Machine Learning Project"
)