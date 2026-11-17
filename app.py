import json
import math
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from shapely.geometry import shape, mapping
from shapely.ops import transform
from pyproj import Transformer

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & THESIS HEADER
# ---------------------------------------------------------
st.set_page_config(
    page_title="Bangkok Urban Wetland Cooling Simulator",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌿 Bangkok Urban Wetland Cooling Performance & Planning Simulator")
st.markdown(
    "**Thesis Prototype:** *Urban Wetland as Cooling Resilient Infrastructure — "
    "Evaluating Cooling Performance and Planning Application in Bangkok (Benjakitti Study Site)*"
)

# ---------------------------------------------------------
# 2. LOAD EMPIRICAL FIELD DATA & WETLAND BOUNDARY
# ---------------------------------------------------------
@st.cache_data
def load_data():
    master = pd.read_csv("bangkok_master_cooling_dataset.csv")
    summary = pd.read_csv("bangkok_sensor_cooling_summary.csv")
    with open("bangkok_wetland_boundary.geojson", "r") as f:
        wetland_gj = json.load(f)
    
    # Merge coordinates into summary table for map plotting
    coords = master[["sensor_id", "latitude", "longitude", "easting_m", "northing_m"]].drop_duplicates()
    summary = summary.merge(coords, on="sensor_id", how="left")
    return master, summary, wetland_gj

try:
    master_df, summary_df, wetland_geojson = load_data()
except FileNotFoundError as e:
    st.error(
        f"Missing required data file: {e}. Please ensure `bangkok_master_cooling_dataset.csv`, "
        "`bangkok_sensor_cooling_summary.csv`, and `bangkok_wetland_boundary.geojson` are in the same folder as `app.py`."
    )
    st.stop()

# ---------------------------------------------------------
# 3. SIDEBAR: URBAN PLANNING & CLIMATE SCENARIO SLIDERS
# ---------------------------------------------------------
st.sidebar.header("🎛️ Planning & Climate Controls")
st.sidebar.caption("Adjust urban design & meteorological parameters to simulate neighborhood cooling.")

# Planning Scenario Sliders
wetland_scale_pct = st.sidebar.slider(
    "Wetland Area Expansion / Reduction (%)",
    min_value=50,
    max_value=150,
    value=100,
    step=10,
    help="100% = Baseline 88.8 ha wetland footprint. Increasing wetland area scales cooling reach and intensity."
)

target_distance_m = st.sidebar.slider(
    "Target Building / Street Distance from Edge (m)",
    min_value=0,
    max_value=800,
    value=150,
    step=25,
    help="Evaluate predicted cooling benefit at a specific urban setback distance."
)

wind_direction_deg = st.sidebar.slider(
    "Prevailing Wind Direction (° Azimuth)",
    min_value=0,
    max_value=360,
    value=231,
    step=5,
    help="Baseline = 231° (Southwesterly monsoon airflow observed during July 21 – Aug 3, 2026)."
)

wind_speed_ms = st.sidebar.slider(
    "Ambient Wind Speed (m/s)",
    min_value=0.5,
    max_value=5.0,
    value=1.9,
    step=0.1,
    help="Baseline mean wind speed = 1.92 m/s."
)

solar_rad_wm2 = st.sidebar.slider(
    "Peak Afternoon Solar Radiation (W/m²)",
    min_value=200,
    max_value=1000,
    value=650,
    step=50,
    help="Higher solar load amplifies evapotranspiration cooling contrast."
)

cooling_threshold_c = st.sidebar.selectbox(
    "Planning Cooling Threshold (°C)",
    options=[0.5, 1.0, 1.5],
    index=0,
    help="Minimum temperature drop required to define the effective Maximum Cooling Distance (MCD)."
)

# ---------------------------------------------------------
# 4. EMPIRICAL COOLING ENGINE (CALIBRATED FROM BANGKOK DATA)
# ---------------------------------------------------------
# Baseline empirical exponential decay from Chapter 1.3:
# DeltaT(d) = 2.097 * exp(-0.00490 * d) - 0.050
area_factor = (wetland_scale_pct / 100.0) ** 0.35
solar_adj = 0.0026 * (solar_rad_wm2 - 650.0)
wind_adj = -0.1884 * (wind_speed_ms - 1.92)

# Peak core cooling intensity
peak_core_dt = max(0.2, (2.097 * area_factor) + solar_adj + wind_adj)
decay_k = 0.00490 / area_factor  # Larger wetlands extend cooling plume farther

def predict_cooling(dist_m, is_downwind=False):
    downwind_bonus = 0.15 if is_downwind else 0.0
    dt = (peak_core_dt + downwind_bonus) * math.exp(-decay_k * dist_m) - 0.050
    return max(0.0, round(dt, 2))

# Calculate Maximum Cooling Distance (MCD) for selected threshold
if peak_core_dt > (cooling_threshold_c + 0.050):
    mcd_meters = round(-math.log((cooling_threshold_c + 0.050) / peak_core_dt) / decay_k, 1)
else:
    mcd_meters = 0.0

# Downwind extended MCD (+15% advection reach along prevailing wind axis)
mcd_downwind_meters = round(mcd_meters * 1.18, 1)

# Predicted cooling at user's target street distance
dt_crosswind_target = predict_cooling(target_distance_m, is_downwind=False)
dt_downwind_target = predict_cooling(target_distance_m, is_downwind=True)

# ---------------------------------------------------------
# 5. TOP KPI METRICS BAR
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label="Simulated Wetland Area",
        value=f"{88.84 * (wetland_scale_pct / 100.0):.1f} ha",
        delta=f"{wetland_scale_pct - 100:+d}% vs. Baseline"
    )
with col2:
    st.metric(
        label="Peak Core Cooling (ΔT)",
        value=f"-{peak_core_dt:.2f} °C",
        delta=f"{peak_core_dt - 2.10:+.2f} °C vs. Observed"
    )
with col3:
    st.metric(
        label=f"Cooling Reach (≥{cooling_threshold_c}°C)",
        value=f"{mcd_meters:.0f} m",
        delta=f"Downwind: {mcd_downwind_meters:.0f} m"
    )
with col4:
    st.metric(
        label=f"Cooling at {target_distance_m}m Setback",
        value=f"-{dt_crosswind_target:.2f} °C",
        delta=f"Downwind: -{dt_downwind_target:.2f} °C"
    )

st.divider()

# ---------------------------------------------------------
# 6. GEOSPATIAL MAP & DISTANCE-DECAY CURVE (2-COLUMN LAYOUT)
# ---------------------------------------------------------
map_col, chart_col = st.columns([1.35, 1.0])

with map_col:
    st.subheader("🗺️ Interactive Spatial Cooling Plume Map (Bangkok)")
    
    # Project wetland polygon to UTM 47N to generate accurate metric buffers
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32647", always_xy=True)
    to_wgs = Transformer.from_crs("EPSG:32647", "EPSG:4326", always_xy=True)
    
    poly_wgs = shape(wetland_geojson["features"][0]["geometry"])
    poly_utm = transform(to_utm.transform, poly_wgs)
    
    # Scale wetland polygon around its centroid if user changed wetland size slider
    if wetland_scale_pct != 100:
        scale_linear = math.sqrt(wetland_scale_pct / 100.0)
        from shapely.affinity import scale
        poly_utm_sim = scale(poly_utm, xfact=scale_linear, yfact=scale_linear, origin="center")
    else:
        poly_utm_sim = poly_utm

    # Create Outer Cooling Buffer (MCD) and Inner High-Intensity Buffer (100m)
    outer_buffer_utm = poly_utm_sim.buffer(mcd_meters) if mcd_meters > 0 else poly_utm_sim
    inner_buffer_utm = poly_utm_sim.buffer(100.0)
    
    # Shift outer buffer slightly downwind to visualize wind advection plume
    downwind_rad = math.radians((wind_direction_deg + 180) % 360)
    advection_shift_m = (mcd_downwind_meters - mcd_meters) * 0.8
    from shapely.affinity import translate
    plume_utm = translate(
        outer_buffer_utm,
        xoff=advection_shift_m * math.sin(downwind_rad),
        yoff=advection_shift_m * math.cos(downwind_rad)
    )

    sim_wetland_wgs = mapping(transform(to_wgs.transform, poly_utm_sim))
    inner_buf_wgs = mapping(transform(to_wgs.transform, inner_buffer_utm))
    plume_wgs = mapping(transform(to_wgs.transform, plume_utm))

    # Initialize Folium Map centered on Benjakitti Wetland
    m = folium.Map(location=[13.7295, 100.5585], zoom_start=15, tiles="CartoDB positron")

    # Layer 1: Simulated Downwind Cooling Plume (MCD Threshold)
    if mcd_meters > 0:
        folium.GeoJson(
            plume_wgs,
            name=f"Effective Cooling Plume (≥{cooling_threshold_c}°C, ~{mcd_meters:.0f}m)",
            style_function=lambda x: {
                "fillColor": "#3182bd",
                "color": "#08519c",
                "weight": 1.5,
                "dashArray": "5, 5",
                "fillOpacity": 0.16,
            },
        ).add_to(m)

    # Layer 2: 100m High-Cooling Buffer
    folium.GeoJson(
        inner_buf_wgs,
        name="100m Near-Park Cooling Zone",
        style_function=lambda x: {
            "fillColor": "#41ab5d",
            "color": "#238b45",
            "weight": 1.5,
            "fillOpacity": 0.22,
        },
    ).add_to(m)

    # Layer 3: Wetland Core Polygon
    folium.GeoJson(
        sim_wetland_wgs,
        name="Urban Wetland Core Boundary",
        style_function=lambda x: {
            "fillColor": "#006d2c",
            "color": "#00441b",
            "weight": 2.5,
            "fillOpacity": 0.55,
        },
    ).add_to(m)

    # Layer 4: 22 Validated Outdoor Field Sensors
    zone_palette = {
        "Core_Wetland": "#006d2c",
        "Near_Buffer_0_150m": "#2b8cbe",
        "Mid_Buffer_150_400m": "#fdae6b",
        "Urban_Reference_400m+": "#d7301f",
    }

    for _, row in summary_df.iterrows():
        color = zone_palette.get(row["zone_type"], "#333333")
        popup_html = (
            f"<b>{row['sensor_id']}</b> ({row['zone_type']})<br>"
            f"Distance to Wetland: <b>{row['distance_to_wetland_m']:.1f} m</b><br>"
            f"Wind Exposure: {row['wind_exposure']}<br>"
            f"Peak Afternoon Temp: <b>{row['peak_T_afternoon']:.2f} °C</b><br>"
            f"Peak Cooling (ΔT): <b>{row['peak_DeltaT_afternoon']:+.2f} °C</b><br>"
            f"Peak Heat Index: {row['peak_HI_afternoon']:.2f} °C"
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=7,
            color="black",
            weight=1.2,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{row['sensor_id']} ({row['distance_to_wetland_m']:.0f}m): ΔT {row['peak_DeltaT_afternoon']:+.2f}°C",
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, width=None, height=500, returned_objects=[])

with chart_col:
    st.subheader("📈 Distance-Decay Cooling Curve")
    distances = np.linspace(0, 800, 160)
    curve_cross = [predict_cooling(d, is_downwind=False) for d in distances]
    curve_down = [predict_cooling(d, is_downwind=True) for d in distances]

    chart_df = pd.DataFrame({
        "Distance from Wetland (m)": distances,
        "Crosswind / Upwind ΔT (°C)": curve_cross,
        "Downwind Corridor ΔT (°C)": curve_down,
    }).set_index("Distance from Wetland (m)")

    st.line_chart(chart_df, height=310)

    # Urban Planning Recommendation Box
    st.info(
        f"**Planning Application Insight (at {target_distance_m}m Setback):**

"
        f"- **Crosswind / General Street:** Expected peak afternoon temperature reduction is **{dt_crosswind_target:.2f} °C**.
"
        f"- **Downwind Ventilation Corridor ({wind_direction_deg}°):** Expected peak reduction reaches **{dt_downwind_target:.2f} °C**.
"
        f"- **Zoning Recommendation:** Maintain open street canyons and low-rise setbacks within **{mcd_meters:.0f} m – {mcd_downwind_meters:.0f} m** of the wetland edge to preserve the ≥{cooling_threshold_c}°C cooling benefit."
    )

# ---------------------------------------------------------
# 7. EXPANDABLE FIELD DATA EXPLORER FOR THESIS APPENDIX
# ---------------------------------------------------------
with st.expander("📊 View Field Sensor Summary Table (22 Validated Outdoor Loggers)"):
    st.dataframe(
        summary_df[[
            "sensor_id", "zone_type", "distance_to_wetland_m", "wind_exposure",
            "mean_T_24h", "peak_T_afternoon", "peak_DeltaT_afternoon", "peak_HI_afternoon"
        ]].sort_values("distance_to_wetland_m"),
        use_container_width=True,
        hide_index=True
    )
