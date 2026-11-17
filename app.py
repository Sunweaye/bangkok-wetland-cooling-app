import json
import math
import os
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from shapely.geometry import shape, mapping, Point
from shapely.ops import transform
from shapely.affinity import scale, translate
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
# 2. BULLETPROOF DATA LOADER (COMPUTES ALL METRICS ON THE FLY)
# ---------------------------------------------------------
def calc_heat_index_c(T_c, RH):
    T_f = T_c * 9.0 / 5.0 + 32.0
    HI_f = (
        -42.379
        + 2.04901523 * T_f
        + 10.14333127 * RH
        - 0.22475541 * T_f * RH
        - 0.00683783 * (T_f ** 2)
        - 0.05481717 * (RH ** 2)
        + 0.00122874 * (T_f ** 2) * RH
        + 0.00085282 * T_f * (RH ** 2)
        - 0.00000199 * (T_f ** 2) * (RH ** 2)
    )
    steadman = 0.5 * (T_f + 61.0 + ((T_f - 68.0) * 1.2) + (RH * 0.094))
    HI_f = np.where(steadman < 80.0, steadman, HI_f)
    return (HI_f - 32.0) * 5.0 / 9.0

@st.cache_data
def load_data():
    # Find whichever master CSV exists in the repo
    csv_candidates = [
        "bangkok_master_cooling_dataset.csv",
        "bangkok_master_dataset_spatial.csv",
        "bangkok_sensors_with_weather.csv"
    ]
    master = None
    for fname in csv_candidates:
        if os.path.exists(fname):
            master = pd.read_csv(fname)
            break
    if master is None:
        raise FileNotFoundError("bangkok_master_cooling_dataset.csv")

    with open("bangkok_wetland_boundary.geojson", "r") as f:
        wetland_gj = json.load(f)

    # Ensure timestamp & hour exist
    master["timestamp"] = pd.to_datetime(master["timestamp"])
    if "hour" not in master.columns:
        master["hour"] = master["timestamp"].dt.hour
    if "is_peak_afternoon" not in master.columns:
        master["is_peak_afternoon"] = master["hour"].between(11, 16)

    # Ensure spatial metrics exist (compute from GeoJSON if missing)
    poly_features = [feat for feat in wetland_gj["features"] if feat["geometry"]["type"] in ["Polygon", "MultiPolygon"]]
    poly_wgs = shape(poly_features[0]["geometry"])
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32647", always_xy=True)
    poly_utm = transform(to_utm.transform, poly_wgs)
    centroid_utm = poly_utm.centroid

    if "distance_to_wetland_m" not in master.columns or "zone_type" not in master.columns or "wind_exposure" not in master.columns:
        unique_pts = master[["sensor_id", "easting_m", "northing_m"]].drop_duplicates()
        spatial_rows = []
        downwind_dir = (231.08 + 180) % 360
        for _, r in unique_pts.iterrows():
            pt = Point(r["easting_m"], r["northing_m"])
            inside = poly_utm.contains(pt) or poly_utm.touches(pt)
            dist_m = 0.0 if inside else round(poly_utm.distance(pt), 1)
            dx = r["easting_m"] - centroid_utm.x
            dy = r["northing_m"] - centroid_utm.y
            bearing = (math.degrees(math.atan2(dx, dy)) + 360) % 360
            ang_diff = abs((bearing - downwind_dir + 180) % 360 - 180)
            if inside:
                w_exp = "Core_Wetland"
                z_type = "Core_Wetland"
            else:
                w_exp = "Downwind_NE" if ang_diff <= 45 else ("Upwind_SW" if ang_diff >= 135 else "Crosswind")
                z_type = "Near_Buffer_0_150m" if dist_m <= 150 else ("Mid_Buffer_150_400m" if dist_m <= 400 else "Urban_Reference_400m+")
            spatial_rows.append({
                "sensor_id": r["sensor_id"],
                "distance_to_wetland_m": dist_m,
                "zone_type": z_type,
                "wind_exposure": w_exp
            })
        spatial_df = pd.DataFrame(spatial_rows)
        for col in ["distance_to_wetland_m", "zone_type", "wind_exposure"]:
            if col in master.columns:
                master = master.drop(columns=[col])
        master = master.merge(spatial_df, on="sensor_id", how="left")

    # Ensure Heat Index exists
    if "heat_index_c" not in master.columns:
        master["heat_index_c"] = calc_heat_index_c(master["air_temp"], master["rel_humidity"]).round(2)

    # Filter out the 2 indoor AC sensors (Sensor_16 & Sensor_18)
    valid_master = master[~master["sensor_id"].isin(["Sensor_16", "Sensor_18"])].copy()

    # Ensure cooling_delta_t_c exists (compute vs Urban_Reference_400m+ if missing)
    if "cooling_delta_t_c" not in valid_master.columns:
        urban_ref = valid_master[valid_master["zone_type"] == "Urban_Reference_400m+"].groupby("timestamp")["air_temp"].mean().rename("urban_ref_temp_c")
        valid_master = valid_master.merge(urban_ref, on="timestamp", how="left")
        valid_master["cooling_delta_t_c"] = (valid_master["urban_ref_temp_c"] - valid_master["air_temp"]).round(2)

    # Build clean 22-sensor summary table
    summary_all = valid_master.groupby(
        ["sensor_id", "zone_type", "distance_to_wetland_m", "wind_exposure", "latitude", "longitude", "easting_m", "northing_m"],
        as_index=False
    ).agg(
        mean_T_24h=("air_temp", "mean"),
        mean_DeltaT_24h=("cooling_delta_t_c", "mean"),
        mean_HI_24h=("heat_index_c", "mean"),
    )

    summary_peak = valid_master[valid_master["is_peak_afternoon"]].groupby(
        "sensor_id", as_index=False
    ).agg(
        peak_T_afternoon=("air_temp", "mean"),
        peak_DeltaT_afternoon=("cooling_delta_t_c", "mean"),
        peak_HI_afternoon=("heat_index_c", "mean"),
    )

    summary = summary_all.merge(summary_peak, on="sensor_id", how="left")
    return valid_master, summary, wetland_gj

try:
    master_df, summary_df, wetland_geojson = load_data()
except FileNotFoundError as e:
    st.error(
        f"Missing required data file: {e}. Please ensure `bangkok_master_cooling_dataset.csv` "
        "and `bangkok_wetland_boundary.geojson` are in the same repository folder as `app.py`."
    )
    st.stop()

# ---------------------------------------------------------
# 3. SIDEBAR: URBAN PLANNING & CLIMATE SCENARIO SLIDERS
# ---------------------------------------------------------
st.sidebar.header("🎛️ Planning & Climate Controls")
st.sidebar.caption("Adjust urban design & meteorological parameters to simulate neighborhood cooling.")

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
area_factor = (wetland_scale_pct / 100.0) ** 0.35
solar_adj = 0.0026 * (solar_rad_wm2 - 650.0)
wind_adj = -0.1884 * (wind_speed_ms - 1.92)

peak_core_dt = max(0.2, (2.097 * area_factor) + solar_adj + wind_adj)
decay_k = 0.00490 / area_factor

def predict_cooling(dist_m, is_downwind=False):
    downwind_bonus = 0.15 if is_downwind else 0.0
    dt = (peak_core_dt + downwind_bonus) * math.exp(-decay_k * dist_m) - 0.050
    return max(0.0, round(dt, 2))

if peak_core_dt > (cooling_threshold_c + 0.050):
    mcd_meters = round(-math.log((cooling_threshold_c + 0.050) / peak_core_dt) / decay_k, 1)
else:
    mcd_meters = 0.0

mcd_downwind_meters = round(mcd_meters * 1.18, 1)

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
    
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32647", always_xy=True)
    to_wgs = Transformer.from_crs("EPSG:32647", "EPSG:4326", always_xy=True)
    
    poly_features = [feat for feat in wetland_geojson["features"] if feat["geometry"]["type"] in ["Polygon", "MultiPolygon"]]
    poly_wgs = shape(poly_features[0]["geometry"])
    poly_utm = transform(to_utm.transform, poly_wgs)
    
    if wetland_scale_pct != 100:
        scale_linear = math.sqrt(wetland_scale_pct / 100.0)
        poly_utm_sim = scale(poly_utm, xfact=scale_linear, yfact=scale_linear, origin="center")
    else:
        poly_utm_sim = poly_utm

    outer_buffer_utm = poly_utm_sim.buffer(mcd_meters) if mcd_meters > 0 else poly_utm_sim
    inner_buffer_utm = poly_utm_sim.buffer(100.0)
    
    downwind_rad = math.radians((wind_direction_deg + 180) % 360)
    advection_shift_m = (mcd_downwind_meters - mcd_meters) * 0.8
    plume_utm = translate(
        outer_buffer_utm,
        xoff=advection_shift_m * math.sin(downwind_rad),
        yoff=advection_shift_m * math.cos(downwind_rad)
    )

    sim_wetland_wgs = mapping(transform(to_wgs.transform, poly_utm_sim))
    inner_buf_wgs = mapping(transform(to_wgs.transform, inner_buffer_utm))
    plume_wgs = mapping(transform(to_wgs.transform, plume_utm))

    m = folium.Map(location=[13.7295, 100.5585], zoom_start=15, tiles="CartoDB positron")

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

    st.info(
        f"**Planning Application Insight (at {target_distance_m}m Setback):**\n\n"
        f"- **Crosswind / General Street:** Expected peak afternoon temperature reduction is **{dt_crosswind_target:.2f} °C**.\n"
        f"- **Downwind Ventilation Corridor ({wind_direction_deg}°):** Expected peak reduction reaches **{dt_downwind_target:.2f} °C**.\n"
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
