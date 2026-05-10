import streamlit as st
import itur
import numpy as np
from scipy.optimize import brentq
from scipy.stats import lognorm
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# ==========================================
# Caching Heavy Computations & API Calls
# ==========================================
@st.cache_data(show_spinner=False)
def get_coordinates(city_name):
    """Fetches latitude and longitude using Geopy."""
    geolocator = Nominatim(user_agent="rain_anomaly_detector_v2")
    try:
        location = geolocator.geocode(city_name, timeout=10)
        if location:
            return location.latitude, location.longitude, location.address
        return None, None, None
    except Exception as e:
        st.error(f"Geocoding error: {e}")
        return None, None, None

@st.cache_data(show_spinner=False)
def calculate_rain_intensity(freq_GHz, fade_depth_dB, distance_km):
    """
    Step 1: Reverses ITU-R P.838 to find the physical Rain Rate (mm/hr) 
    required to cause the observed fade, using robust root-finding.
    """
    gamma_target = fade_depth_dB / distance_km

    def specific_attenuation_error(R_test):
        # Use the universally public specific_attenuation function
        gamma_test = itur.models.itu838.specific_attenuation(
            f=freq_GHz, R=R_test, el=0, tau=90
        )
        
        # Safely extract value if astropy unit is returned
        gamma_val = getattr(gamma_test, 'value', gamma_test)
        
        # We want the difference to be 0
        return gamma_val - gamma_target

    try:
        # Search for a rain rate between 0.01 mm/hr and 1000.0 mm/hr
        r_result = brentq(specific_attenuation_error, a=0.01, b=1000.0)
        return float(r_result)
    except ValueError:
        # If the fade is so incredibly deep it requires > 1000 mm/hr of rain,
        # it is mathematically an extreme anomaly. We cap it at 1000.
        return 1000.0

@st.cache_data(show_spinner=False)
def estimate_annual_probability(lat, lon, freq_GHz, distance_km, fade_depth_dB):
    """
    Step 2: Uses Brent's root-finding method to reverse ITU-R P.530 and 
    find the annual probability of this fade depth occurring.
    """
    def attenuation_error(p_test):
        A_test = itur.models.itu530.rain_attenuation(
            lat, lon, distance_km, freq_GHz, el=0, p=p_test, tau=90
        )
        A_val = getattr(A_test, 'value', A_test)
        return A_val - fade_depth_dB

    try:
        # Search between 0.00001% (extremely rare) and 10.0% (extremely common)
        p_result = brentq(attenuation_error, a=0.00001, b=10.0)
        return float(p_result)
    except ValueError:
        # If it fails, the fade is outside normal atmospheric boundaries
        # We test the 0.00001% boundary to see if the fade is mathematically too deep
        test_deep = attenuation_error(0.00001)
        if test_deep < 0: 
            return 0.000001 # Fade is deeper than the 0.00001% limit
        return 10.0 # Fade is shallower than the 10% limit

@st.cache_data(show_spinner=False)
def duration_survival_probability(rain_rate_mmhr, duration_minutes):
    """
    Step 3: Meteorological Log-Normal Survival Model.
    Calculates the probability that a storm of intensity R lasts longer than T.
    """
    # Empirical Inverse Power Law for Mean Rain Duration
    # Higher intensity = exponentially shorter mean duration
    mean_duration = 100.0 * (rain_rate_mmhr ** -0.6)
    
    # Standard deviation for meteorological rain events is roughly 1.0
    sigma = 1.0 
    
    # Calculate Log-Normal Survival Function (1 - CDF)
    p_survival = lognorm.sf(duration_minutes, s=sigma, scale=mean_duration)
    
    return float(p_survival), float(mean_duration)

# ==========================================
# Streamlit UI
# ==========================================
st.set_page_config(page_title="Deep Fade Anomaly Detector", layout="wide")

st.title("📡 RSL Deep Fade Anomaly Detector")
st.markdown("This tool validates RSL telemetry by separating the **radio physics** from the **weather statistics**. It calculates the physical rain required for a fade, checks if your climate supports it, and tests if the duration is physically possible.")

# --- Sidebar Inputs ---
st.sidebar.header("1. Link Configuration")
city_input = st.sidebar.text_input("Location (City)", value="Tel Aviv, Israel")
freq_GHz = st.sidebar.number_input("Carrier Frequency (GHz)", min_value=1.0, max_value=100.0, value=18.0, step=1.0)
distance_km = st.sidebar.number_input("Link Length (km)", min_value=0.1, max_value=50.0, value=5.0, step=0.5)

st.sidebar.header("2. RSL Event Telemetry")
fade_depth = st.sidebar.number_input("Fade Depth (dB Drop)", min_value=1.0, max_value=100.0, value=25.0, step=1.0)
duration_minutes = st.sidebar.number_input("Event Duration (Minutes)", min_value=1.0, max_value=1440.0, value=45.0, step=1.0)

st.sidebar.markdown("---")
anomaly_threshold = st.sidebar.slider("Anomaly Threshold (%)", 0.001, 5.0, 0.1, format="%.3f%%")

# --- Execution Logic ---
if st.sidebar.button("Analyze Event", type="primary", use_container_width=True):
    with st.spinner("Geocoding Location..."):
        lat, lon, address = get_coordinates(city_input)
        
    if lat is None:
        st.error("Location not found. Please try a different query.")
        st.stop()
        
    st.caption(f"📍 **Location Profile:** {address} (Lat: {lat:.4f}, Lon: {lon:.4f})")
    
    col1, col2, col3 = st.columns(3)
    
    # STEP 1: Rain Intensity
    with st.spinner("Calculating physical rain constraints..."):
        r_intensity = calculate_rain_intensity(freq_GHz, fade_depth, distance_km)
    
    with col1:
        st.subheader("Step 1: Physics")
        st.metric("Equivalent Rain Rate", f"{r_intensity:.1f} mm/hr")
        st.info(f"A {fade_depth}dB drop at {freq_GHz}GHz over {distance_km}km requires a storm core of **{r_intensity:.1f} mm/hr**.")

    # STEP 2: Annual Probability
    with st.spinner("Querying ITU-R environmental models..."):
        annual_prob = estimate_annual_probability(lat, lon, freq_GHz, distance_km, fade_depth)
    
    with col2:
        st.subheader("Step 2: Climate")
        if annual_prob < 0.0001:
            st.metric("Annual Probability", "< 0.0001%")
            st.error("This fade depth is almost mathematically impossible for this region.")
        else:
            st.metric("Annual Probability", f"{annual_prob:.4f}%")
            st.info(f"This region is expected to experience fades of {fade_depth}dB roughly **{(annual_prob/100)*525600:.0f} minutes** per year.")

    # STEP 3: Duration Survival
    with st.spinner("Running meteorological survival model..."):
        surv_prob, mean_dur = duration_survival_probability(r_intensity, duration_minutes)
        surv_prob_pct = surv_prob * 100.0
        
    with col3:
        st.subheader("Step 3: Duration")
        st.metric("Probability of Duration", f"{surv_prob_pct:.4f}%")
        st.info(f"An intense {r_intensity:.1f} mm/hr storm averages only **{mean_dur:.1f} minutes**. The chance of it lasting {duration_minutes} minutes is **{surv_prob_pct:.4f}%**.")

    # --- Final Classification ---
    st.divider()
    st.subheader("Final Classification")
    
    is_anomaly = False
    reasons = []
    
    if annual_prob < (anomaly_threshold / 100.0): # Convert threshold to raw float for annual check
        is_anomaly = True
        reasons.append("The fade depth is too severe for the local climate.")
    
    if surv_prob_pct < anomaly_threshold:
        is_anomaly = True
        reasons.append(f"The event duration ({duration_minutes} mins) physically violates the fluid dynamics of a {r_intensity:.1f} mm/hr storm.")

    if is_anomaly:
        st.error("🚨 **CLASSIFICATION: ANOMALY (NON-RAIN EVENT)**")
        for r in reasons:
            st.write(f"- {r}")
        st.warning("**Recommended Action:** Inspect telemetry for Wet Radome Attenuation (WAA), hardware failure, or antenna misalignment.")
    else:
        st.success("🌧️ **CLASSIFICATION: NORMAL RAIN FADE**")
        st.write("This event fits within the physical and climatological boundaries of the ITU-R and meteorological models.")
