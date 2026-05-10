import streamlit as st
import itur
import numpy as np
from scipy.optimize import brentq
from scipy.stats import lognorm
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# ==========================================
# Embedded Spatial Climate Database
# Sourced from COST 210, Crane Models, and Regional Met Studies
# ==========================================
CLIMATE_DB = {
    "Tel Aviv, Israel": {"lat": 32.0853, "lon": 34.7818, "a": 95.0, "b": 0.63, "type": "Mediterranean"},
    "Barcelona, Spain": {"lat": 41.3851, "lon": 2.1734, "a": 92.0, "b": 0.64, "type": "Mediterranean"},
    "Dubai, UAE": {"lat": 25.2048, "lon": 55.2708, "a": 105.0, "b": 0.60, "type": "Arid / Subtropical"},
    "Berlin, Germany": {"lat": 52.5200, "lon": 13.4050, "a": 88.0, "b": 0.66, "type": "Temperate Continental"},
    "Warsaw, Poland": {"lat": 52.2297, "lon": 21.0122, "a": 85.0, "b": 0.67, "type": "Temperate Continental"},
    "New York, USA": {"lat": 40.7128, "lon": -74.0060, "a": 115.0, "b": 0.61, "type": "Humid Subtropical / Continental"},
    "Dallas, USA": {"lat": 32.7767, "lon": -96.7970, "a": 120.0, "b": 0.59, "type": "Humid Subtropical"},
    "Tokyo, Japan": {"lat": 35.6762, "lon": 139.6503, "a": 125.0, "b": 0.58, "type": "Humid Subtropical"},
    "Seoul, South Korea": {"lat": 37.5665, "lon": 126.9780, "a": 130.0, "b": 0.56, "type": "Humid Continental / Monsoon"},
    "New Delhi, India": {"lat": 28.6139, "lon": 77.2090, "a": 140.0, "b": 0.54, "type": "Monsoon"},
    "Kuala Lumpur, Malaysia": {"lat": 3.1390, "lon": 101.6869, "a": 145.0, "b": 0.52, "type": "Equatorial Rainforest"},
    "London, UK": {"lat": 51.5074, "lon": -0.1278, "a": 82.0, "b": 0.68, "type": "Temperate Oceanic"}
}

# ==========================================
# Caching Heavy Computations & API Calls
# ==========================================
@st.cache_data(show_spinner=False)
def get_coordinates_and_climate(city_name):
    """Fetches coordinates and finds the closest meteorological parameters."""
    geolocator = Nominatim(user_agent="mmwave_anomaly_detector")
    try:
        location = geolocator.geocode(city_name, timeout=10)
        if not location:
            return None, None, None, None
            
        target_coords = (location.latitude, location.longitude)
        
        # Geodesic Nearest Neighbor Search
        closest_city = None
        min_distance = float('inf')
        
        for city, data in CLIMATE_DB.items():
            db_coords = (data['lat'], data['lon'])
            dist_km = geodesic(target_coords, db_coords).km
            if dist_km < min_distance:
                min_distance = dist_km
                closest_city = city
                
        climate_data = CLIMATE_DB[closest_city]
        climate_data['reference_city'] = closest_city
        climate_data['distance_km'] = min_distance
        
        return location.latitude, location.longitude, location.address, climate_data
        
    except Exception as e:
        st.error(f"Geocoding error: {e}")
        return None, None, None, None

@st.cache_data(show_spinner=False)
def calculate_rain_intensity(freq_GHz, fade_depth_dB, distance_km):
    """Step 1: Reverses ITU-R P.838 to find physical Rain Rate."""
    gamma_target = fade_depth_dB / distance_km

    def specific_attenuation_error(R_test):
        gamma_test = itur.models.itu838.rain_specific_attenuation(
            f=freq_GHz, R=R_test, el=0, tau=90
        )
        gamma_val = getattr(gamma_test, 'value', gamma_test)
        return gamma_val - gamma_target

    try:
        r_result = brentq(specific_attenuation_error, a=0.01, b=1000.0)
        return float(r_result)
    except ValueError:
        return 1000.0

@st.cache_data(show_spinner=False)
def estimate_annual_probability(lat, lon, freq_GHz, distance_km, fade_depth_dB):
    """Step 2: Uses ITU-R P.530 to find annual likelihood of fade depth."""
    def attenuation_error(p_test):
        A_test = itur.models.itu530.rain_attenuation(
            lat, lon, distance_km, freq_GHz, el=0, p=p_test, tau=90
        )
        A_val = getattr(A_test, 'value', A_test)
        return A_val - fade_depth_dB

    try:
        p_result = brentq(attenuation_error, a=0.00001, b=10.0)
        return float(p_result)
    except ValueError:
        test_deep = attenuation_error(0.00001)
        if test_deep < 0: return 0.000001
        return 10.0

@st.cache_data(show_spinner=False)
def duration_survival_probability(rain_rate_mmhr, duration_minutes, a_param, b_param):
    """
    Step 3: Meteorological Log-Normal Survival Model.
    Uses dynamically localized a and b parameters.
    """
    mean_duration = a_param * (rain_rate_mmhr ** -b_param)
    sigma = 1.0 
    p_survival = lognorm.sf(duration_minutes, s=sigma, scale=mean_duration)
    return float(p_survival), float(mean_duration)

# ==========================================
# Streamlit UI
# ==========================================
st.set_page_config(page_title="mmWave Deep Fade Detector", layout="wide")

st.title("📡 mmWave Deep Fade Anomaly Detector")
st.markdown("Validates mmWave RSL telemetry using ITU-R scattering limits and spatially-aware meteorological decay models.")

# --- Sidebar Inputs ---
st.sidebar.header("1. Link Configuration")
city_input = st.sidebar.text_input("Location (City)", value="Ramat Gan, Israel")
freq_GHz = st.sidebar.number_input("Carrier Frequency (GHz)", min_value=1.0, max_value=100.0, value=80.0, step=1.0)
distance_km = st.sidebar.number_input("Link Length (km)", min_value=0.1, max_value=50.0, value=3.0, step=0.1)

st.sidebar.header("2. RSL Event Telemetry")
fade_depth = st.sidebar.number_input("Fade Depth (dB Drop)", min_value=1.0, max_value=100.0, value=25.0, step=1.0)
duration_minutes = st.sidebar.number_input("Event Duration (Minutes)", min_value=1.0, max_value=1440.0, value=30.0, step=1.0)

st.sidebar.markdown("---")

st.sidebar.header("3. Detection Thresholds")

# Hardcoded or Dropdown SLA target
annual_sla_target = st.sidebar.selectbox(
    "Link Availability Target (Annual)", 
    options=[0.1, 0.01, 0.001, 0.0001], 
    index=2, 
    format_func=lambda x: f"{x}% ({(x/100)*525600:.0f} mins/yr)"
)

# Adjustable slider for duration confidence
survival_confidence = st.sidebar.slider(
    "Duration Anomaly Confidence limit", 
    min_value=0.001, max_value=5.0, value=1.0, step=0.05, format="%.3f%%",
    help="Triggers an anomaly if the chance of the storm lasting this long drops below this percentage."
)


#anomaly_threshold = st.sidebar.slider("Anomaly Threshold (%)", 0.001, 5.0, 0.1, format="%.3f%%")

# --- Execution Logic ---
if st.sidebar.button("Analyze Event", type="primary", use_container_width=True):
    with st.spinner("Geocoding & Locating Climate Profile..."):
        lat, lon, address, climate = get_coordinates_and_climate(city_input)
        
    if lat is None:
        st.error("Location not found. Please try a different query.")
        st.stop()
        
    st.caption(f"📍 **Geocoded:** {address} (Lat: {lat:.4f}, Lon: {lon:.4f})")
    st.caption(f"🌍 **Climate Anchor:** {climate['type']} (Nearest DB Node: {climate['reference_city']}, {climate['distance_km']:.0f}km away) → a={climate['a']}, b={climate['b']}")
    
    col1, col2, col3 = st.columns(3)
    
    # STEP 1
    with st.spinner("Calculating physical rain constraints..."):
        r_intensity = calculate_rain_intensity(freq_GHz, fade_depth, distance_km)
    with col1:
        st.subheader("Step 1: Physics")
        st.metric("Equivalent Rain Rate", f"{r_intensity:.1f} mm/hr")
        st.info(f"A {fade_depth}dB drop at {freq_GHz}GHz over {distance_km}km requires a storm core of **{r_intensity:.1f} mm/hr**.")

    # STEP 2
    with st.spinner("Querying ITU-R environmental models..."):
        annual_prob = estimate_annual_probability(lat, lon, freq_GHz, distance_km, fade_depth)
    with col2:
        st.subheader("Step 2: Climate Limits")
        if annual_prob < 0.0001:
            st.metric("Annual Probability", "< 0.0001%")
            st.error("This fade depth is statistically impossible for this region.")
        else:
            st.metric("Annual Probability", f"{annual_prob:.4f}%")
            st.info(f"This region typically experiences fades of {fade_depth}dB roughly **{(annual_prob/100)*525600:.0f} minutes** per year.")

    # STEP 3
    with st.spinner("Running spatially-aware survival model..."):
        surv_prob, mean_dur = duration_survival_probability(r_intensity, duration_minutes, climate['a'], climate['b'])
        surv_prob_pct = surv_prob * 100.0
    with col3:
        st.subheader("Step 3: Duration")
        st.metric("Probability of Duration", f"{surv_prob_pct:.4f}%")
        st.info(f"Using the {climate['type']} profile, an intense {r_intensity:.1f} mm/hr storm averages **{mean_dur:.1f} minutes**. The chance of it lasting {duration_minutes} minutes is **{surv_prob_pct:.4f}%**.")

    # --- Final Classification ---
    st.divider()
    st.subheader("Final Classification")
    
    is_anomaly = False
    reasons = []
    
    # 1. Check severity against the Link SLA
    if annual_prob < annual_sla_target:
        is_anomaly = True
        reasons.append(f"Severity Violation: The fade depth requires a storm rarer than the link's {annual_sla_target}% design limit.")
    
    # 2. Check duration against the User Confidence Threshold
    if surv_prob_pct < survival_confidence:
        is_anomaly = True
        reasons.append(f"Duration Violation: The chance of this storm surviving for {duration_minutes} mins is {surv_prob_pct:.4f}%, which falls below your {survival_confidence}% confidence limit.")
    
    if is_anomaly:
        st.error("🚨 **CLASSIFICATION: ANOMALY (NON-RAIN EVENT)**")
        for r in reasons:
            st.write(f"- {r}")
        st.warning("**Recommended Action:** Inspect telemetry for Wet Radome Attenuation (WAA), transceiver hardware failure, or mechanical misalignment.")
    else:
        st.success("🌧️ **CLASSIFICATION: NORMAL RAIN FADE**")
        st.write("This event fits entirely within the electromagnetic scattering limits and regional meteorological decay curves.")
