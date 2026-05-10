import streamlit as st
import itur
from geopy.geocoders import Nominatim

# ==========================================
# Caching Heavy Computations & API Calls
# ==========================================
@st.cache_data(show_spinner=False)
def get_coordinates(city_name):
    """Fetches latitude and longitude for a given city name using Geopy."""
    geolocator = Nominatim(user_agent="rain_anomaly_detector_app")
    try:
        location = geolocator.geocode(city_name)
        if location:
            return location.latitude, location.longitude, location.address
        else:
            return None, None, None
    except Exception as e:
        st.error(f"Geocoding error: {e}")
        return None, None, None

@st.cache_data(show_spinner=False)
def check_rain_probability_itur(lat, lon, freq_GHz, fade_depth_dB, duration_sec):
    """
    Calculates the probability that a rain event maintains a specific 
    fade depth for a given duration using ITU-R P.1623.
    """
    try:
        # el=0 for terrestrial link, p=90 for Vertical polarization
        prob_exceeded = itur.models.itu1623.fade_duration_probability(
            lat=lat, 
            lon=lon, 
            f=freq_GHz, 
            el=0, 
            p=90, 
            D=duration_sec, 
            A=fade_depth_dB
        )
        return float(prob_exceeded)
    except Exception as e:
        st.error(f"ITU-R Calculation Error: {e}")
        return None

# ==========================================
# Streamlit UI
# ==========================================
st.set_page_config(page_title="RSL Anomaly Detector", layout="centered")

st.title("📡 RSL Drop Classifier: Rain vs. Anomaly")
st.markdown("""
This tool uses the **ITU-R P.1623** standard to evaluate whether a specific Received Signal Level (RSL) drop physically matches the profile of a local rain cell, or if it should be flagged as a Wet Radome / Hardware Anomaly.
""")

st.divider()

# --- Input Form ---
with st.container():
    st.subheader("Event Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        city_input = st.text_input("City/Location", value="Tel Aviv, Israel")
        freq_GHz = st.number_input("Carrier Frequency (GHz)", min_value=1.0, max_value=100.0, value=18.0, step=1.0)
        
    with col2:
        fade_depth = st.number_input("Fade Depth (dB Drop)", min_value=1.0, max_value=80.0, value=25.0, step=1.0)
        duration_minutes = st.number_input("Fade Duration (Minutes)", min_value=0.1, max_value=1440.0, value=10.0, step=1.0)

    # Allow user to set the strictness of the anomaly detector
    st.markdown("---")
    threshold_pct = st.slider(
        "Anomaly Threshold (%) - If probability falls below this, it is flagged as an anomaly.", 
        min_value=0.001, max_value=5.0, value=1.0, step=0.01, format="%.3f%%"
    )

# --- Execution Logic ---
if st.button("Evaluate RSL Drop", type="primary", use_container_width=True):
    duration_seconds = duration_minutes * 60.0
    
    with st.spinner("Locating coordinates..."):
        lat, lon, address = get_coordinates(city_input)
        
    if lat is None:
        st.error("Location not found. Please try a different city name.")
    else:
        st.caption(f"📍 **Location:** {address} (Lat: {lat:.4f}, Lon: {lon:.4f})")
        
        with st.spinner("Calculating ITU-R P.1623 Survival Probability..."):
            prob = check_rain_probability_itur(lat, lon, freq_GHz, fade_depth, duration_seconds)
            
        if prob is not None:
            prob_pct = prob * 100.0
            
            st.divider()
            st.subheader("Analysis Results")
            
            # Display the raw metric
            col_metric, col_label = st.columns([1, 1])
            with col_metric:
                st.metric(label="Likelihood of Rain Duration", value=f"{prob_pct:.4f}%")
                
            # Decision Logic / Classification Label
            with col_label:
                if prob_pct >= threshold_pct:
                    st.success("🌧️ **CLASSIFICATION: RAIN**")
                    st.info(f"This duration is statistically normal for a {fade_depth} dB rain fade in this region.")
                else:
                    st.error("🚨 **CLASSIFICATION: ANOMALY**")
                    st.warning(f"Mathematical limit exceeded. Rain rarely maintains a {fade_depth} dB drop for {duration_minutes} minutes here. \n\n**Likely Causes:** Wet Radome (WAA), Hardware Degradation, or Antenna Misalignment.")
