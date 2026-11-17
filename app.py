import json
import math
import os
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from shapely.geometry import shape, mapping, Point, Polygon, LineString
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
# 3. BANGKOK BOUNDARY, VEGETATION ZONES & COOLING CORRIDORS
# ---------------------------------------------------------
# Bangkok Metropolitan Administration (BMA) Outer Boundary Polygon (WGS84 lon, lat)
BANGKOK_BMA_BOUNDARY_COORDS = [
    [100.328, 13.785], [100.345, 13.865], [100.435, 13.925], [100.545, 13.955],
    [100.640, 13.945], [100.745, 13.925], [100.885, 13.895], [100.938, 13.835],
    [100.915, 13.735], [100.865, 13.685], [100.765, 13.675], [100.685, 13.655],
    [100.605, 13.665], [100.575, 13.685], [100.535, 13.675], [100.495, 13.645],
    [100.445, 13.595], [100.415, 13.525], [100.445, 13.495], [100.495, 13.515],
    [100.475, 13.575], [100.415, 13.625], [100.355, 13.665], [100.328, 13.785]
]

# Existing & Proposed Urban Wetlands / Retention Parks in Bangkok
BANGKOK_WETLAND_SITES = [
    {
        "name": "Benjakitti Forest Park & Wetland (Calibration Site)",
        "district": "Khlong Toei",
        "status": "Existing",
        "area_ha": 88.84,
        "use_geojson": True,
        "coords_wgs84": None,
    },
    {
        "name": "Lumphini Park & Retention Lake",
        "district": "Pathum Wan",
        "status": "Existing",
        "area_ha": 57.6,
        "use_geojson": False,
        "coords_wgs84": [
            [100.5378, 13.7345], [100.5446, 13.7342],
            [100.5450, 13.7282], [100.5382, 13.7285], [100.5378, 13.7345]
        ],
    },
    {
        "name": "Makkasan Wetland & Retention Basin",
        "district": "Ratchathewi",
        "status": "Existing / Upgradeable",
        "area_ha": 42.0,
        "use_geojson": False,
        "coords_wgs84": [
            [100.5485, 13.7542], [100.5585, 13.7540],
            [100.5585, 13.7508], [100.5485, 13.7510], [100.5485, 13.7542]
        ],
    },
    {
        "name": "Wachirabenchathat (Rot Fai) & Chatuchak Wetland",
        "district": "Chatuchak",
        "status": "Existing",
        "area_ha": 75.0,
        "use_geojson": False,
        "coords_wgs84": [
            [100.5500, 13.8165], [100.5585, 13.8160],
            [100.5580, 13.8065], [100.5505, 13.8070], [100.5500, 13.8165]
        ],
    },
    {
        "name": "Rama IX Royal Park & Retention Reservoir",
        "district": "Prawet",
        "status": "Existing",
        "area_ha": 80.0,
        "use_geojson": False,
        "coords_wgs84": [
            [100.6590, 13.6920], [100.6680, 13.6920],
            [100.6680, 13.6840], [100.6590, 13.6840], [100.6590, 13.6920]
        ],
    },
    {
        "name": "Nong Bon Water Sports & Retention Basin",
        "district": "Prawet",
        "status": "Existing",
        "area_ha": 135.0,
        "use_geojson": False,
        "coords_wgs84": [
            [100.6745, 13.6940], [100.6865, 13.6940],
            [100.6865, 13.6830], [100.6745, 13.6830], [100.6745, 13.6940]
        ],
    },
    {
        "name": "Bang Kachao / Sri Nakhon Khuean Khan Green Lung",
        "district": "Southern River Bend",
        "status": "Existing",
        "area_ha": 320.0,
        "use_geojson": False,
        "coords_wgs84": [
            [100.5540, 13.7045], [100.5730, 13.7045],
            [100.5730, 13.6885], [100.5540, 13.6885], [100.5540, 13.7045]
        ],
    },
    {
        "name": "Future Vision: Khlong Toei Port Blue-Green Wetland Park",
        "district": "Khlong Toei Riverfront",
        "status": "Future Vision",
        "area_ha": 95.0,
        "use_geojson": False,
        "coords_wgs84": [
            [100.5660, 13.7115], [100.5810, 13.7115],
            [100.5810, 13.7035], [100.5660, 13.7035], [100.5660, 13.7115]
        ],
    },
    {
        "name": "Future Vision: Eastern Kaem Ling Retention Wetland (Lat Krabang)",
        "district": "Lat Krabang / Min Buri",
        "status": "Future Vision",
        "area_ha": 160.0,
        "use_geojson": False,
        "coords_wgs84": [
            [100.7550, 13.7650], [100.7720, 13.7650],
            [100.7720, 13.7520], [100.7550, 13.7520], [100.7550, 13.7650]
        ],
    },
]

# Urban Vegetation & Canopy Zones (Existing Urban Forests, Agricultural Green Belts, Tree Canopies)
VEGETATION_ZONES = [
    {
        "name": "Bang Kachao Tropical Urban Forest & Orchard Canopy",
        "type": "Existing Dense Vegetation",
        "canopy_pct": 78,
        "coords": [
            [100.546, 13.711], [100.584, 13.711], [100.588, 13.682],
            [100.550, 13.678], [100.542, 13.695], [100.546, 13.711]
        ]
    },
    {
        "name": "Thonburi Western Agricultural & Canal Orchard Belt",
        "type": "Existing Peri-Urban Vegetation",
        "canopy_pct": 62,
        "coords": [
            [100.345, 13.810], [100.425, 13.810], [100.425, 13.720],
            [100.350, 13.720], [100.345, 13.810]
        ]
    },
    {
        "name": "Nong Chok / Min Buri Eastern Green Belt & Paddy",
        "type": "Existing Peri-Urban Vegetation",
        "canopy_pct": 58,
        "coords": [
            [100.780, 13.880], [100.905, 13.870], [100.890, 13.760],
            [100.775, 13.770], [100.780, 13.880]
        ]
    },
    {
        "name": "Chatuchak–Vachirabenchathat–Queen Sirikit Urban Forest Cluster",
        "type": "Existing Urban Canopy",
        "canopy_pct": 68,
        "coords": [
            [100.547, 13.819], [100.562, 13.819], [100.562, 13.802],
            [100.547, 13.802], [100.547, 13.819]
        ]
    },
    {
        "name": "Future Vision: Central CBD Canopy Ring (Lumphini–Benjakitti–Makkasan)",
        "type": "Future Urban Forestry Vision",
        "canopy_pct": 45,
        "coords": [
            [100.535, 13.756], [100.564, 13.756], [100.566, 13.724],
            [100.535, 13.724], [100.535, 13.756]
        ]
    },
]

# Cooling & Ventilation Corridors (Aligned with Prevailing 231° SW Monsoon & Canal Network)
COOLING_CORRIDORS = [
    {
        "name": "Corridor 1: Benjakitti – Lumphini Green Bridge Skywalk & Tree Spine",
        "status": "Existing + Future Expansion",
        "width_m": 180,
        "bonus_dt": 0.45,
        "coords": [[100.5414, 13.7314], [100.5500, 13.7305], [100.5585, 13.7295]],
    },
    {
        "name": "Corridor 2: Bang Kachao – Benjakitti – Makkasan SW Monsoon Wind Axis (231°)",
        "status": "Future Strategic Cooling Corridor",
        "width_m": 320,
        "bonus_dt": 0.65,
        "coords": [
            [100.5635, 13.6965], [100.5610, 13.7130], [100.5585, 13.7295],
            [100.5560, 13.7410], [100.5535, 13.7525]
        ],
    },
    {
        "name": "Corridor 3: Khlong Saen Saep Blue-Green Canal Ventilation Spine",
        "status": "Future Canal Riparian Corridor",
        "width_m": 220,
        "bonus_dt": 0.50,
        "coords": [
            [100.5250, 13.7510], [100.5535, 13.7525], [100.6050, 13.7580],
            [100.6650, 13.7650], [100.7350, 13.7750]
        ],
    },
    {
        "name": "Corridor 4: Chao Phraya Riverfront Breeze & Riparian Parkways",
        "status": "Existing Primary Blue Axis",
        "width_m": 350,
        "bonus_dt": 0.70,
        "coords": [
            [100.5150, 13.7950], [100.5020, 13.7550], [100.5120, 13.7220],
            [100.5350, 13.6950], [100.5635, 13.6965]
        ],
    },
    {
        "name": "Corridor 5: Rama IX – Nong Bon – Lat Krabang Eastern Retention Link",
        "status": "Future Eastern Floodplain Corridor",
        "width_m": 280,
        "bonus_dt": 0.55,
        "coords": [
            [100.6635, 13.6880], [100.6805, 13.6885], [100.7220, 13.7220], [100.7635, 13.7585]
        ],
    },
]

# ---------------------------------------------------------
# 4. SIDEBAR: SIMULATION SCOPE, VISION PRESET & SLIDERS
# ---------------------------------------------------------
st.sidebar.header("🧭 1. Planning Vision & Scope")

sim_mode = st.sidebar.radio(
    "Map & Analysis Scale:",
    options=[
        "Bangkok-Wide Vision (Boundary + Vegetation + Corridors)",
        "Benjakitti Empirical Site (22 Field Sensors Validation)"
    ],
    index=0
)

vision_scenario = st.sidebar.selectbox(
    "Select Planning Scenario:",
    options=[
        "Existing Situation (2026 Baseline)",
        "Future Vision 2035: Blue-Green Network + Cooling Corridors",
        "Future Stress Test: Extreme Heatwave + Future Corridors"
    ],
    index=1
)

# Apply preset values based on selected scenario while keeping sliders interactive
if vision_scenario == "Existing Situation (2026 Baseline)":
    preset_scale = 100
    preset_solar = 650
    preset_wind = 1.9
    include_future_nodes = False
    corridor_boost_on = False
elif vision_scenario == "Future Vision 2035: Blue-Green Network + Cooling Corridors":
    preset_scale = 130
    preset_solar = 650
    preset_wind = 2.1
    include_future_nodes = True
    corridor_boost_on = True
else:
    preset_scale = 130
    preset_solar = 950
    preset_wind = 1.2
    include_future_nodes = True
    corridor_boost_on = True

st.sidebar.divider()
st.sidebar.header("🗺️ 2. Map Layer Toggles")
show_bma_boundary = st.sidebar.checkbox("Show Bangkok Administrative Boundary", value=True)
show_vegetation = st.sidebar.checkbox("Show Vegetation & Urban Canopy Zones", value=True)
show_corridors = st.sidebar.checkbox("Show Blue-Green Cooling Corridors", value=True)

st.sidebar.divider()
st.sidebar.header("🎛️ 3. Fine-Tune Simulation Sliders")

wetland_scale_pct = st.sidebar.slider(
    "Wetland Area Expansion / Reduction (%)",
    min_value=50,
    max_value=150,
    value=preset_scale,
    step=10
)

target_distance_m = st.sidebar.slider(
    "Target Urban Street Setback Distance (m)",
    min_value=0,
    max_value=800,
    value=150,
    step=25
)

wind_direction_deg = st.sidebar.slider(
    "Prevailing Wind Direction (° Azimuth)",
    min_value=0,
    max_value=360,
    value=231,
    step=5,
    help="231° = Southwesterly monsoon airflow observed in Bangkok."
)

wind_speed_ms = st.sidebar.slider(
    "Ambient Wind Speed (m/s)",
    min_value=0.5,
    max_value=5.0,
    value=preset_wind,
    step=0.1
)

solar_rad_wm2 = st.sidebar.slider(
    "Peak Afternoon Solar Radiation (W/m²)",
    min_value=200,
    max_value=1000,
    value=preset_solar,
    step=50
)

cooling_threshold_c = st.sidebar.selectbox(
    "Planning Cooling Threshold (°C)",
    options=[0.5, 1.0, 1.5],
    index=0
)

# ---------------------------------------------------------
# 5. EMPIRICAL COOLING ENGINE
# ---------------------------------------------------------
def compute_site_metrics(base_area_ha, scale_pct, solar_wm2, wind_ms, threshold_c, setback_m, corridor_active=False):
    sim_area_ha = base_area_ha * (scale_pct / 100.0)
    area_factor = (sim_area_ha / 88.84) ** 0.35
    solar_adj = 0.0026 * (solar_wm2 - 650.0)
    wind_adj = -0.1884 * (wind_ms - 1.92)
    corridor_dt_bonus = 0.25 if corridor_active else 0.0

    core_dt = max(0.2, (2.097 * area_factor) + solar_adj + wind_adj + corridor_dt_bonus)
    k = 0.00490 / (area_factor * (1.12 if corridor_active else 1.0))

    def pred(d, downwind=False):
        bonus = 0.15 if downwind else 0.0
        val = (core_dt + bonus) * math.exp(-k * d) - 0.050
        return max(0.0, round(val, 2))

    if core_dt > (threshold_c + 0.050):
        mcd = round(-math.log((threshold_c + 0.050) / core_dt) / k, 1)
    else:
        mcd = 0.0

    mcd_down = round(mcd * (1.25 if corridor_active else 1.18), 1)
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
bma_poly_wgs = Polygon(BANGKOK_BMA_BOUNDARY_COORDS)
bma_poly_utm = transform(to_utm.transform, bma_poly_wgs)

# Helper to add common Bangkok Boundary, Vegetation, and Corridor layers to any Folium map
def add_bangkok_planning_layers(fmap):
    # 1. Bangkok Administrative Boundary
    if show_bma_boundary:
        folium.GeoJson(
            mapping(bma_poly_wgs),
            name="Bangkok Administrative Boundary (BMA)",
            style_function=lambda x: {
                "fillColor": "#f7fbff",
                "color": "#d95f02",
                "weight": 3.0,
                "dashArray": "8, 5",
                "fillOpacity": 0.04,
            },
            tooltip="Bangkok Metropolitan Administration (BMA) Boundary (~1,569 km²)"
        ).add_to(fmap)

    # 2. Vegetation & Urban Canopy Zones
    if show_vegetation:
        for vz in VEGETATION_ZONES:
            if "Future" in vz["type"] and not include_future_nodes:
                continue
            is_future = "Future" in vz["type"]
            v_poly = Polygon(vz["coords"])
            folium.GeoJson(
                mapping(v_poly),
                name=f"Vegetation: {vz['name']}",
                style_function=lambda x, fut=is_future: {
                    "fillColor": "#78c679" if not fut else "#addd8e",
                    "color": "#238443",
                    "weight": 1.2,
                    "dashArray": "4, 4" if fut else None,
                    "fillOpacity": 0.25 if not fut else 0.20,
                },
                tooltip=f"{vz['name']} ({vz['type']} | Canopy: {vz['canopy_pct']}%)"
            ).add_to(fmap)

    # 3. Cooling & Ventilation Corridors
    if show_corridors:
        for corr in COOLING_CORRIDORS:
            if "Future" in corr["status"] and not corridor_boost_on:
                continue
            is_future_c = "Future" in corr["status"]
            line_wgs = LineString(corr["coords"])
            line_utm = transform(to_utm.transform, line_wgs)
            buf_utm = line_utm.buffer(corr["width_m"])
            buf_wgs = transform(to_wgs.transform, buf_utm)

            folium.GeoJson(
                mapping(buf_wgs),
                style_function=lambda x, fut=is_future_c: {
                    "fillColor": "#00bcd4" if fut else "#2b8cbe",
                    "color": "#006064" if fut else "#045a8d",
                    "weight": 1.5,
                    "dashArray": "6, 4" if fut else None,
                    "fillOpacity": 0.28,
                },
                tooltip=f"{corr['name']} ({corr['status']} | Extra Cooling: -{corr['bonus_dt']:.2f}°C)"
            ).add_to(fmap)

# =========================================================
# VIEW A: BANGKOK-WIDE VISION (EXISTING VS FUTURE SCENARIO)
# =========================================================
if sim_mode.startswith("Bangkok-Wide"):
    # Compute Existing Baseline vs Current Selected Vision for direct comparison
    active_sites = [
        s for s in BANGKOK_WETLAND_SITES
        if s["status"] != "Future Vision" or include_future_nodes
    ]

    city_rows = []
    plume_geoms_utm = []
    core_geoms_utm = []
    downwind_rad = math.radians((wind_direction_deg + 180) % 360)

    m_city = folium.Map(location=[13.7450, 100.6050], zoom_start=11, tiles="OpenStreetMap", name="Bangkok Street Map")
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite Aerial View",
        overlay=False,
        control=True
    ).add_to(m_city)

    add_bangkok_planning_layers(m_city)

    for site in active_sites:
        s_res = compute_site_metrics(
            site["area_ha"], wetland_scale_pct, solar_rad_wm2, wind_speed_ms,
            cooling_threshold_c, target_distance_m, corridor_active=corridor_boost_on
        )
        city_rows.append({
            "Wetland / Retention Node": site["name"],
            "District": site["district"],
            "Status": site["status"],
            "Simulated Area (ha)": s_res["sim_area_ha"],
            "Peak Core ΔT (°C)": -s_res["core_dt"],
            f"Crosswind Reach ≥{cooling_threshold_c}°C (m)": s_res["mcd_cross"],
            f"Downwind Reach ≥{cooling_threshold_c}°C (m)": s_res["mcd_down"],
            f"ΔT at {target_distance_m}m Downwind (°C)": -s_res["dt_setback_down"],
        })

        p_wgs = benjakitti_poly_wgs if site["use_geojson"] else Polygon(site["coords_wgs84"])
        p_utm = transform(to_utm.transform, p_wgs)
        if wetland_scale_pct != 100:
            s_lin = math.sqrt(wetland_scale_pct / 100.0)
            p_utm = scale(p_utm, xfact=s_lin, yfact=s_lin, origin="center")

        core_geoms_utm.append(p_utm)
        outer_b = p_utm.buffer(s_res["mcd_cross"]) if s_res["mcd_cross"] > 0 else p_utm
        shift_m = (s_res["mcd_down"] - s_res["mcd_cross"]) * 0.85
        plume_b = translate(outer_b, xoff=shift_m * math.sin(downwind_rad), yoff=shift_m * math.cos(downwind_rad))
        plume_geoms_utm.append(plume_b)

        is_future_site = site["status"] == "Future Vision"
        folium.GeoJson(
            mapping(transform(to_wgs.transform, plume_b)),
            style_function=lambda x: {"fillColor": "#3182bd", "color": "#08519c", "weight": 1.2, "dashArray": "4, 4", "fillOpacity": 0.22},
            tooltip=f"{site['name']} Cooling Plume: {s_res['mcd_cross']:.0f}m – {s_res['mcd_down']:.0f}m"
        ).add_to(m_city)

        folium.GeoJson(
            mapping(transform(to_wgs.transform, p_utm)),
            style_function=lambda x, fut=is_future_site: {
                "fillColor": "#006d2c" if not fut else "#00a86b",
                "color": "#00441b",
                "weight": 2.2,
                "dashArray": "5, 3" if fut else None,
                "fillOpacity": 0.72,
            },
            popup=folium.Popup(
                f"<b>{site['name']}</b> ({site['status']})<br>"
                f"Simulated Area: <b>{s_res['sim_area_ha']:.1f} ha</b><br>"
                f"Peak Core ΔT: <b>-{s_res['core_dt']:.2f} °C</b><br>"
                f"Downwind Cooling Reach: <b>{s_res['mcd_down']:.0f} m</b>",
                max_width=270
            )
        ).add_to(m_city)

    # Include corridor buffers in total city cooled area when corridors are active
    all_cooled_geoms = list(plume_geoms_utm)
    if show_corridors:
        for corr in COOLING_CORRIDORS:
            if "Future" in corr["status"] and not corridor_boost_on:
                continue
            c_utm = transform(to_utm.transform, LineString(corr["coords"])).buffer(corr["width_m"])
            all_cooled_geoms.append(c_utm)

    merged_core_ha = unary_union(core_geoms_utm).area / 10000.0
    merged_cooled_ha = max(0.0, (unary_union(all_cooled_geoms).area / 10000.0) - merged_core_ha)

    # Calculate Vegetation Area inside BMA
    active_veg = [v for v in VEGETATION_ZONES if "Future" not in v["type"] or include_future_nodes]
    veg_geoms_utm = [transform(to_utm.transform, Polygon(v["coords"])) for v in active_veg]
    merged_veg_ha = unary_union(veg_geoms_utm).area / 10000.0

    # Baseline reference values for delta display (Existing 2026: 798.4 ha wetland, ~1,185 ha cooled urban zone)
    baseline_wetland_ha = 798.4
    baseline_cooled_ha = 1185.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Wetland Core Area",
            f"{merged_core_ha:.1f} ha",
            f"{merged_core_ha - baseline_wetland_ha:+.1f} ha vs. Existing 2026"
        )
    with col2:
        st.metric(
            "Urban Vegetation & Canopy Zones",
            f"{merged_veg_ha:,.0f} ha",
            f"{len(active_veg)} Green Belts Mapped"
        )
    with col3:
        st.metric(
            "Net Cooled Urban Zone (≥0.5°C)",
            f"{merged_cooled_ha:,.1f} ha",
            f"{merged_cooled_ha - baseline_cooled_ha:+.1f} ha vs. Existing 2026"
        )
    with col4:
        avg_core_dt = np.mean([r["Peak Core ΔT (°C)"] for r in city_rows])
        st.metric(
            "Network Mean Core Cooling",
            f"{avg_core_dt:.2f} °C",
            f"{'Future Corridors Active' if corridor_boost_on else 'Existing Baseline'}"
        )

    st.divider()

    # Map Legend Banner
    st.markdown(
        "**Map Legend:** &nbsp; "
        "🟧 **Dashed Orange:** Bangkok (BMA) Boundary &nbsp;|&nbsp; "
        "🟩 **Dark Green:** Wetland Cores &nbsp;|&nbsp; "
        "🌿 **Light Green:** Urban Vegetation & Canopy Belts &nbsp;|&nbsp; "
        "🟦 **Cyan / Blue Channels:** Cooling & Wind Corridors + Plumes"
    )

    folium.LayerControl(collapsed=True).add_to(m_city)
    st_folium(m_city, width=None, height=540, returned_objects=[])

    # Existing vs. Future Vision Comparison Summary Box
    st.subheader("📊 Existing Situation vs. Future Vision Comparison (Bangkok-Wide)")
    comp_df = pd.DataFrame([
        {
            "Planning Stage": "Existing Situation (2026 Baseline)",
            "Active Wetland Nodes": "7 Existing Parks & Basins (798.4 ha)",
            "Vegetation & Canopy Strategy": "4 Existing Urban/Peri-Urban Canopy Belts",
            "Cooling Corridors": "Chao Phraya River + Partial Benjakitti–Lumphini Link",
            "Peak Core Cooling Range": "-1.63 °C to -3.30 °C",
            "Net Cooled Surrounding Urban Area": "~1,185 ha"
        },
        {
            "Planning Stage": "Future Vision 2035 (Blue-Green + Corridors)",
            "Active Wetland Nodes": "9 Nodes (+30% Expansion + Khlong Toei & Lat Krabang Wetlands: 1,369 ha)",
            "Vegetation & Canopy Strategy": "Central CBD Canopy Ring + Expanded Riparian Tree Belts",
            "Cooling Corridors": "5 Integrated Blue-Green Wind Corridors (231° SW Monsoon Aligned)",
            "Peak Core Cooling Range": "-2.07 °C to -3.90 °C",
            "Net Cooled Surrounding Urban Area": "~3,450+ ha (+191% Urban Cooling Coverage)"
        }
    ])
    st.table(comp_df)

    st.subheader("📋 Node-by-Node Bangkok Cooling Performance Table")
    city_df = pd.DataFrame(city_rows)
    st.dataframe(city_df, use_container_width=True, hide_index=True)

    st.download_button(
        label="⬇️ Download Bangkok Scenario Results (CSV)",
        data=city_df.to_csv(index=False).encode("utf-8"),
        file_name="bangkok_existing_vs_future_cooling_scenarios.csv",
        mime="text/csv"
    )

# =========================================================
# VIEW B: BENJAKITTI EMPIRICAL SITE (22 SENSORS VALIDATION)
# =========================================================
else:
    res = compute_site_metrics(
        88.84, wetland_scale_pct, solar_rad_wm2, wind_speed_ms,
        cooling_threshold_c, target_distance_m, corridor_active=corridor_boost_on
    )

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
        st.subheader("🗺️ Benjakitti Empirical Site, Vegetation & Corridors")
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

        m = folium.Map(location=[13.7295, 100.5585], zoom_start=14, tiles="OpenStreetMap", name="Bangkok Street Map")
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            name="Satellite Aerial View",
            overlay=False,
            control=True
        ).add_to(m)

        add_bangkok_planning_layers(m)

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

        folium.LayerControl(collapsed=True).add_to(m)
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
