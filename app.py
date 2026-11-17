import json
import math
import os
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from shapely.geometry import shape, mapping, Point, Polygon
from shapely.ops import transform, unary_union
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
    "Evaluating Cooling Performance and Planning Application in Bangkok*"
)

# ---------------------------------------------------------
# 2. HELPER FUNCTIONS & EMPIRICAL DATA LOADER
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

    master["timestamp"] = pd.to_datetime(master["timestamp"])
    if "hour" not in master.columns:
        master["hour"] = master["timestamp"].dt.hour
    if "is_peak_afternoon" not in master.columns:
        master["is_peak_afternoon"] = master["hour"].between(11, 16)

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

    if "heat_index_c" not in master.columns:
        master["heat_index_c"] = calc_heat_index_c(master["air_temp"], master["rel_humidity"]).round(2)

    valid_master = master[~master["sensor_id"].isin(["Sensor_16", "Sensor_18"])].copy()

    if "cooling_delta_t_c" not in valid_master.columns:
        urban_ref = valid_master[valid_master["zone_type"] == "Urban_Reference_400m+"].groupby("timestamp")["air_temp"].mean().rename("urban_ref_temp_c")
        valid_master = valid_master.merge(urban_ref, on="timestamp", how="left")
        valid_master["cooling_delta_t_c"] = (valid_master["urban_ref_temp_c"] - valid_master["air_temp"]).round(2)

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
# 3. BANGKOK-WIDE WETLAND NETWORK INVENTORY (CITY-SCALE MODE)
# ---------------------------------------------------------
BANGKOK_WETLAND_SITES = [
    {
        "name": "Benjakitti Forest Park & Wetland (Calibration Site)",
        "district": "Khlong Toei",
        "typology": "Urban Forest Wetland",
        "area_ha": 88.84,
        "lat": 13.7295,
        "lon": 100.5585,
        "use_geojson": True,
        "coords_wgs84": None,
    },
    {
        "name": "Lumphini Park & Retention Lake",
        "district": "Pathum Wan",
        "typology": "District Park & Lake",
        "area_ha": 57.6,
        "lat": 13.7314,
        "lon": 100.5414,
        "use_geojson": False,
        "coords_wgs84": [
            [100.5378, 13.7345], [100.5446, 13.7342],
            [100.5450, 13.7282], [100.5382, 13.7285], [100.5378, 13.7345]
        ],
    },
    {
        "name": "Makkasan Wetland & Retention Basin",
        "district": "Ratchathewi",
        "typology": "Rail-Corridor Retention Wetland",
        "area_ha": 42.0,
        "lat": 13.7525,
        "lon": 100.5535,
        "use_geojson": False,
        "coords_wgs84": [
            [100.5485, 13.7542], [100.5585, 13.7540],
            [100.5585, 13.7508], [100.5485, 13.7510], [100.5485, 13.7542]
        ],
    },
    {
        "name": "Wachirabenchathat (Rot Fai) & Chatuchak Wetland",
        "district": "Chatuchak",
        "typology": "Northern District Park & Wetland",
        "area_ha": 75.0,
        "lat": 13.8115,
        "lon": 100.5545,
        "use_geojson": False,
        "coords_wgs84": [
            [100.5500, 13.8165], [100.5585, 13.8160],
            [100.5580, 13.8065], [100.5505, 13.8070], [100.5500, 13.8165]
        ],
    },
    {
        "name": "Rama IX Royal Park & Retention Reservoir",
        "district": "Prawet",
        "typology": "Eastern Retention Park",
        "area_ha": 80.0,
        "lat": 13.6880,
        "lon": 100.6635,
        "use_geojson": False,
        "coords_wgs84": [
            [100.6590, 13.6920], [100.6680, 13.6920],
            [100.6680, 13.6840], [100.6590, 13.6840], [100.6590, 13.6920]
        ],
    },
    {
        "name": "Nong Bon Water Sports & Retention Basin",
        "district": "Prawet",
        "typology": "Peri-Urban Retention Basin",
        "area_ha": 135.0,
        "lat": 13.6885,
        "lon": 100.6805,
        "use_geojson": False,
        "coords_wgs84": [
            [100.6745, 13.6940], [100.6865, 13.6940],
            [100.6865, 13.6830], [100.6745, 13.6830], [100.6745, 13.6940]
        ],
    },
    {
        "name": "Bang Kachao / Sri Nakhon Khuean Khan Wetland Core",
        "district": "Phra Pradaeng (River Bend)",
        "typology": "Mega Blue-Green Floodplain",
        "area_ha": 320.0,
        "lat": 13.6965,
        "lon": 100.5635,
        "use_geojson": False,
        "coords_wgs84": [
            [100.5540, 13.7045], [100.5730, 13.7045],
            [100.5730, 13.6885], [100.5540, 13.6885], [100.5540, 13.7045]
        ],
    },
]

# ---------------------------------------------------------
# 4. SIDEBAR: SIMULATION MODE & PLANNING CONTROLS
# ---------------------------------------------------------
st.sidebar.header("🧭 Simulation Scope")
sim_mode = st.sidebar.radio(
    "Select Planning Scale:",
    options=[
        "1. Benjakitti Empirical Site (Field Validation)",
        "2. Bangkok-Wide Blue-Green Network (City Scale)"
    ],
    index=0
)

st.sidebar.divider()
st.sidebar.header("🎛️ Planning & Climate Controls")

wetland_scale_pct = st.sidebar.slider(
    "Wetland Area Expansion / Reduction (%)",
    min_value=50,
    max_value=150,
    value=100,
    step=10,
    help="100% = Current baseline wetland footprint. Simulates blue-green expansion or encroachment."
)

target_distance_m = st.sidebar.slider(
    "Target Building / Street Setback Distance (m)",
    min_value=0,
    max_value=800,
    value=150,
    step=25,
    help="Evaluate predicted cooling benefit at a specific urban setback distance."
)

season_preset = st.sidebar.selectbox(
    "Bangkok Monsoon Wind Preset",
    options=[
        "Custom Wind Angle",
        "Southwest Monsoon (Wet/Hot Season: 231° SW)",
        "Northeast Monsoon (Cool/Dry Season: 45° NE)"
    ],
    index=0
)

if season_preset == "Southwest Monsoon (Wet/Hot Season: 231° SW)":
    default_wind_dir = 231
elif season_preset == "Northeast Monsoon (Cool/Dry Season: 45° NE)":
    default_wind_dir = 45
else:
    default_wind_dir = 231

wind_direction_deg = st.sidebar.slider(
    "Prevailing Wind Direction (° Azimuth)",
    min_value=0,
    max_value=360,
    value=default_wind_dir,
    step=5,
    help="231° = Southwesterly flow observed during July 21 – Aug 3, 2026."
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
# 5. EMPIRICAL COOLING ENGINE (CALIBRATED FROM BANGKOK DATA)
# ---------------------------------------------------------
def compute_site_metrics(base_area_ha, scale_pct, solar_wm2, wind_ms, threshold_c, setback_m):
    sim_area_ha = base_area_ha * (scale_pct / 100.0)
    area_factor = (sim_area_ha / 88.84) ** 0.35
    solar_adj = 0.0026 * (solar_wm2 - 650.0)
    wind_adj = -0.1884 * (wind_ms - 1.92)

    core_dt = max(0.2, (2.097 * area_factor) + solar_adj + wind_adj)
    k = 0.00490 / area_factor

    def pred(d, downwind=False):
        bonus = 0.15 if downwind else 0.0
        val = (core_dt + bonus) * math.exp(-k * d) - 0.050
        return max(0.0, round(val, 2))

    if core_dt > (threshold_c + 0.050):
        mcd = round(-math.log((threshold_c + 0.050) / core_dt) / k, 1)
    else:
        mcd = 0.0

    mcd_down = round(mcd * 1.18, 1)
    return {
        "sim_area_ha": round(sim_area_ha, 1),
        "core_dt": round(core_dt, 2),
        "decay_k": k,
        "mcd_cross": mcd,
        "mcd_down": mcd_down,
        "dt_setback_cross": pred(setback_m, False),
        "dt_setback_down": pred(setback_m, True),
        "predict_fn": pred
    }

to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32647", always_xy=True)
to_wgs = Transformer.from_crs("EPSG:32647", "EPSG:4326", always_xy=True)
poly_features = [feat for feat in wetland_geojson["features"] if feat["geometry"]["type"] in ["Polygon", "MultiPolygon"]]
benjakitti_poly_wgs = shape(poly_features[0]["geometry"])

# =========================================================
# MODE 1: BENJAKITTI EMPIRICAL SITE (FIELD VALIDATION)
# =========================================================
if sim_mode.startswith("1."):
    res = compute_site_metrics(88.84, wetland_scale_pct, solar_rad_wm2, wind_speed_ms, cooling_threshold_c, target_distance_m)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Simulated Wetland Area", f"{res['sim_area_ha']:.1f} ha", f"{wetland_scale_pct - 100:+d}% vs. Baseline")
    with col2:
        st.metric("Peak Core Cooling (ΔT)", f"-{res['core_dt']:.2f} °C", f"{res['core_dt'] - 2.10:+.2f} °C vs. Observed")
    with col3:
        st.metric(f"Cooling Reach (≥{cooling_threshold_c}°C)", f"{res['mcd_cross']:.0f} m", f"Downwind: {res['mcd_down']:.0f} m")
    with col4:
        st.metric(f"Cooling at {target_distance_m}m Setback", f"-{res['dt_setback_cross']:.2f} °C", f"Downwind: -{res['dt_setback_down']:.2f} °C")

    st.divider()
    map_col, chart_col = st.columns([1.35, 1.0])

    with map_col:
        st.subheader("🗺️ Benjakitti Spatial Cooling Plume & Field Loggers")
        poly_utm = transform(to_utm.transform, benjakitti_poly_wgs)
        if wetland_scale_pct != 100:
            scale_lin = math.sqrt(wetland_scale_pct / 100.0)
            poly_utm_sim = scale(poly_utm, xfact=scale_lin, yfact=scale_lin, origin="center")
        else:
            poly_utm_sim = poly_utm

        outer_buf = poly_utm_sim.buffer(res["mcd_cross"]) if res["mcd_cross"] > 0 else poly_utm_sim
        inner_buf = poly_utm_sim.buffer(100.0)

        downwind_rad = math.radians((wind_direction_deg + 180) % 360)
        shift_m = (res["mcd_down"] - res["mcd_cross"]) * 0.8
        plume_utm = translate(outer_buf, xoff=shift_m * math.sin(downwind_rad), yoff=shift_m * math.cos(downwind_rad))

        m = folium.Map(location=[13.7295, 100.5585], zoom_start=15, tiles="OpenStreetMap", name="Bangkok Street Map")
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            name="Satellite Aerial View",
            overlay=False,
            control=True
        ).add_to(m)

        if res["mcd_cross"] > 0:
            folium.GeoJson(
                mapping(transform(to_wgs.transform, plume_utm)),
                name=f"Effective Cooling Plume (≥{cooling_threshold_c}°C, ~{res['mcd_cross']:.0f}m)",
                style_function=lambda x: {"fillColor": "#3182bd", "color": "#08519c", "weight": 1.5, "dashArray": "5, 5", "fillOpacity": 0.18},
            ).add_to(m)

        folium.GeoJson(
            mapping(transform(to_wgs.transform, inner_buf)),
            name="100m Near-Park Cooling Zone",
            style_function=lambda x: {"fillColor": "#41ab5d", "color": "#238b45", "weight": 1.5, "fillOpacity": 0.24},
        ).add_to(m)

        folium.GeoJson(
            mapping(transform(to_wgs.transform, poly_utm_sim)),
            name="Urban Wetland Core Boundary",
            style_function=lambda x: {"fillColor": "#006d2c", "color": "#00441b", "weight": 2.5, "fillOpacity": 0.55},
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
        chart_df = pd.DataFrame({
            "Distance from Wetland (m)": distances,
            "Crosswind / Upwind ΔT (°C)": [res["predict_fn"](d, False) for d in distances],
            "Downwind Corridor ΔT (°C)": [res["predict_fn"](d, True) for d in distances],
        }).set_index("Distance from Wetland (m)")
        st.line_chart(chart_df, height=310)

        st.info(
            f"**Site Planning Insight (at {target_distance_m}m Setback):**\n\n"
            f"- **Crosswind / General Street:** Expected peak afternoon cooling is **{res['dt_setback_cross']:.2f} °C**.\n"
            f"- **Downwind Ventilation Corridor ({wind_direction_deg}°):** Expected peak cooling reaches **{res['dt_setback_down']:.2f} °C**.\n"
            f"- **Zoning Guideline:** Preserve open street canyons and low-rise step-downs within **{res['mcd_cross']:.0f} m – {res['mcd_down']:.0f} m** of the wetland boundary."
        )

    with st.expander("📊 View Field Sensor Summary Table (22 Validated Outdoor Loggers)"):
        st.dataframe(
            summary_df[[
                "sensor_id", "zone_type", "distance_to_wetland_m", "wind_exposure",
                "mean_T_24h", "peak_T_afternoon", "peak_DeltaT_afternoon", "peak_HI_afternoon"
            ]].sort_values("distance_to_wetland_m"),
            use_container_width=True,
            hide_index=True
        )

# =========================================================
# MODE 2: BANGKOK-WIDE BLUE-GREEN NETWORK (CITY SCALE)
# =========================================================
else:
    st.info(
        "**City-Scale Extrapolation Mode:** Applies the Benjakitti empirical distance-decay scaling model across "
        "7 major Bangkok urban wetlands and retention basins to evaluate city-wide cooling corridors and green-link connectivity."
    )

    city_rows = []
    plume_geoms_utm = []
    core_geoms_utm = []

    m_city = folium.Map(location=[13.7350, 100.5950], zoom_start=12, tiles="OpenStreetMap", name="Bangkok Street Map")
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite Aerial View",
        overlay=False,
        control=True
    ).add_to(m_city)

    downwind_rad = math.radians((wind_direction_deg + 180) % 360)

    for site in BANGKOK_WETLAND_SITES:
        s_res = compute_site_metrics(
            site["area_ha"], wetland_scale_pct, solar_rad_wm2, wind_speed_ms, cooling_threshold_c, target_distance_m
        )
        city_rows.append({
            "Bangkok Wetland / Retention Site": site["name"],
            "District": site["district"],
            "Typology": site["typology"],
            "Simulated Area (ha)": s_res["sim_area_ha"],
            "Peak Core ΔT (°C)": -s_res["core_dt"],
            f"Crosswind Reach ≥{cooling_threshold_c}°C (m)": s_res["mcd_cross"],
            f"Downwind Reach ≥{cooling_threshold_c}°C (m)": s_res["mcd_down"],
            f"ΔT at {target_distance_m}m Downwind (°C)": -s_res["dt_setback_down"],
        })

        # Construct polygon in UTM 47N
        if site["use_geojson"]:
            p_wgs = benjakitti_poly_wgs
        else:
            p_wgs = Polygon(site["coords_wgs84"])
        p_utm = transform(to_utm.transform, p_wgs)

        if wetland_scale_pct != 100:
            s_lin = math.sqrt(wetland_scale_pct / 100.0)
            p_utm = scale(p_utm, xfact=s_lin, yfact=s_lin, origin="center")

        core_geoms_utm.append(p_utm)
        outer_b = p_utm.buffer(s_res["mcd_cross"]) if s_res["mcd_cross"] > 0 else p_utm
        shift_m = (s_res["mcd_down"] - s_res["mcd_cross"]) * 0.8
        plume_b = translate(outer_b, xoff=shift_m * math.sin(downwind_rad), yoff=shift_m * math.cos(downwind_rad))
        plume_geoms_utm.append(plume_b)

        # Add Plume & Core layers to City Map
        folium.GeoJson(
            mapping(transform(to_wgs.transform, plume_b)),
            style_function=lambda x: {"fillColor": "#3182bd", "color": "#08519c", "weight": 1.2, "dashArray": "4, 4", "fillOpacity": 0.22},
            tooltip=f"{site['name']} Cooling Plume: {s_res['mcd_cross']:.0f}m–{s_res['mcd_down']:.0f}m"
        ).add_to(m_city)

        folium.GeoJson(
            mapping(transform(to_wgs.transform, p_utm)),
            style_function=lambda x: {"fillColor": "#006d2c", "color": "#00441b", "weight": 2.0, "fillOpacity": 0.65},
            popup=folium.Popup(
                f"<b>{site['name']}</b><br>"
                f"Area: <b>{s_res['sim_area_ha']:.1f} ha</b><br>"
                f"Peak Core ΔT: <b>-{s_res['core_dt']:.2f} °C</b><br>"
                f"Downwind Reach: <b>{s_res['mcd_down']:.0f} m</b>",
                max_width=260
            )
        ).add_to(m_city)

    # Calculate total merged city-wide cooled urban footprint (accounting for overlapping plumes!)
    merged_core_ha = unary_union(core_geoms_utm).area / 10000.0
    merged_plume_ha = unary_union(plume_geoms_utm).area / 10000.0
    net_cooled_urban_ha = max(0.0, merged_plume_ha - merged_core_ha)

    # Check if Benjakitti and Lumphini cooling plumes merge into a continuous Green Bridge corridor!
    benja_lumphini_connected = plume_geoms_utm[0].intersects(plume_geoms_utm[1])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Network Wetland Area", f"{merged_core_ha:.1f} ha", f"{len(BANGKOK_WETLAND_SITES)} Major Nodes")
    with col2:
        st.metric("Net Cooled Urban Surrounding Zone", f"{net_cooled_urban_ha:.1f} ha", f"≥{cooling_threshold_c}°C Relief")
    with col3:
        st.metric("Benjakitti–Lumphini Plume Link", "CONNECTED ✅" if benja_lumphini_connected else "SEPARATED ⚠️", "Central CBD Corridor")
    with col4:
        avg_core_dt = np.mean([r["Peak Core ΔT (°C)"] for r in city_rows])
        st.metric("Network Mean Core Cooling", f"{avg_core_dt:.2f} °C", f"Wind: {wind_direction_deg}° @ {wind_speed_ms} m/s")

    st.divider()
    st.subheader("🗺️ Bangkok Metropolitan Blue-Green Cooling Network Map")
    folium.LayerControl(collapsed=False).add_to(m_city)
    st_folium(m_city, width=None, height=520, returned_objects=[])

    st.subheader("📋 City-Wide Scenario Comparison Matrix (Exportable for Thesis)")
    city_df = pd.DataFrame(city_rows)
    st.dataframe(city_df, use_container_width=True, hide_index=True)

    csv_export = city_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Bangkok-Wide Scenario Table (CSV)",
        data=csv_export,
        file_name="bangkok_citywide_wetland_cooling_scenarios.csv",
        mime="text/csv"
    )
