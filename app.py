import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Airbnb Smart Pricing Prediction",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
with open("price_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

df = pd.read_csv("data/AB_NYC_2019.csv")

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.big-title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: white;
}

.subtitle {
    text-align: center;
    color: #B0B0B0;
    font-size: 18px;
}

.prediction-box {
    background: linear-gradient(90deg,#11998e,#38ef7d);
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    color: white;
    margin-top: 20px;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
}

.section-header {
    color: white;
    font-size: 22px;
    font-weight: bold;
    margin-top: 30px;
    margin-bottom: 10px;
}

.info-card {
    background-color: #1A1F2B;
    border: 1px solid #2A2F3B;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    color: white;
}

.info-card h4 {
    color: #B0B0B0;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.info-card h2 {
    color: white;
    font-size: 28px;
    margin: 0;
}

.budget-card {
    background: linear-gradient(135deg, #2c3e50, #4a6178);
}

.recommended-card {
    background: linear-gradient(135deg, #11998e, #38ef7d);
}

.peak-card {
    background: linear-gradient(135deg, #c31432, #240b36);
}

.diff-positive {
    color: #38ef7d;
}

.diff-negative {
    color: #ff6b6b;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:

    st.title("📊 Project Info")

    st.markdown("---")

    st.info("""
### Airbnb Smart Pricing Prediction

Predict Airbnb listing prices using Machine Learning.

### Model Used
Random Forest Regressor

### Dataset
AB_NYC_2019

### Tech Stack
- Python
- Pandas
- Scikit-Learn
- Streamlit
""")

    st.markdown("---")

    st.success("✅ Model Loaded Successfully")
    st.markdown("---")

st.markdown("### 📈 Model Performance")

st.metric("MAE", "69.69")
st.metric("CV MAE", "83.07")

st.markdown("---")

st.markdown("### 📊 Dataset Statistics")

st.write(f"Rows: {len(df):,}")
st.write(f"Columns: {len(df.columns)}")

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    '<div class="big-title">🏠 Airbnb Smart Pricing Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict the optimal Airbnb listing price using Machine Learning</div>',
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# INPUTS
# -----------------------------
col1, col2 = st.columns(2)

with col1:

    neighbourhood = st.selectbox(
        "📍 Neighbourhood",
        [
            "Bronx",
            "Brooklyn",
            "Manhattan",
            "Queens",
            "Staten Island"
        ]
    )

    minimum_nights = st.number_input(
        "🌙 Minimum Nights",
        min_value=1,
        value=1
    )

    number_of_reviews = st.number_input(
        "⭐ Number of Reviews",
        min_value=0,
        value=0
    )

with col2:

    room_type = st.selectbox(
        "🛏 Room Type",
        [
            "Entire home/apt",
            "Private room",
            "Shared room"
        ]
    )

    host_listings_count = st.number_input(
        "🏠 Host Listings Count",
        min_value=1,
        value=1
    )

    availability = st.number_input(
        "📅 Availability (Days Per Year)",
        min_value=0,
        max_value=365,
        value=0
    )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔮 Predict Airbnb Price"):

    input_df = pd.DataFrame({
        "minimum_nights": [minimum_nights],
        "number_of_reviews": [number_of_reviews],
        "calculated_host_listings_count": [host_listings_count],
        "availability_365": [availability],
        "neighbourhood_group": [neighbourhood],
        "room_type": [room_type]
    })

    # Convert to dummy variables
    input_df = pd.get_dummies(input_df)

    # Add missing columns
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Arrange columns exactly like training
    input_df = input_df[feature_columns]

    prediction = model.predict(input_df)[0]

    # -----------------------------
    # MAIN PREDICTION DISPLAY
    # -----------------------------
    st.success("✅ Prediction Generated Successfully")

    st.markdown(
        f"""
        <div class="prediction-box">
            <h2>💰 Predicted Airbnb Price</h2>
            <h1>${prediction:.2f}</h1>
            <h3>per night</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------
    # DYNAMIC PRICING RECOMMENDATION
    # -----------------------------
    low_price = prediction * 0.90
    recommended_price = prediction
    high_price = prediction * 1.10

    st.markdown('<div class="section-header">📈 Dynamic Pricing Recommendations</div>', unsafe_allow_html=True)

    dp_col1, dp_col2, dp_col3 = st.columns(3)

    with dp_col1:
        st.markdown(
            f"""
            <div class="info-card budget-card">
                <h4>Budget Price</h4>
                <h2>${low_price:.2f}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with dp_col2:
        st.markdown(
            f"""
            <div class="info-card recommended-card">
                <h4>Recommended Price</h4>
                <h2>${recommended_price:.2f}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with dp_col3:
        st.markdown(
            f"""
            <div class="info-card peak-card">
                <h4>Peak Demand Price</h4>
                <h2>${high_price:.2f}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -----------------------------
    # COMPETITION COMPARISON
    # -----------------------------
    st.markdown('<div class="section-header">🏘️ Competition Comparison</div>', unsafe_allow_html=True)

    area_listings = df[df["neighbourhood_group"] == neighbourhood]

    if len(area_listings) > 0 and "price" in area_listings.columns:
        area_avg_price = area_listings["price"].mean()
        price_difference = prediction - area_avg_price
        diff_class = "diff-positive" if price_difference >= 0 else "diff-negative"
        diff_sign = "+" if price_difference >= 0 else ""

        cc_col1, cc_col2, cc_col3 = st.columns(3)

        with cc_col1:
            st.markdown(
                f"""
                <div class="info-card">
                    <h4>Area Average Price</h4>
                    <h2>${area_avg_price:.2f}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        with cc_col2:
            st.markdown(
                f"""
                <div class="info-card">
                    <h4>Your Predicted Price</h4>
                    <h2>${prediction:.2f}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        with cc_col3:
            st.markdown(
                f"""
                <div class="info-card">
                    <h4>Difference</h4>
                    <h2 class="{diff_class}">{diff_sign}${price_difference:.2f}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.warning("⚠️ No comparable listings found for this neighbourhood in the dataset.")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("---")

st.caption(
    "Built by Namrata N. S | Data Analytics & Machine Learning Project"
)