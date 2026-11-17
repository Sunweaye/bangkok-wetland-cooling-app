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

    bma_districts_gj = None
    if os.path.exists("bangkok_bma_districts.geojson"):
        with open("bangkok_bma_districts.geojson", "r") as f:
            bma_districts_gj = json.load(f)

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
    return valid_master, summary, wetland_gj, bma_districts_gj

try:
    master_df, summary_df, wetland_geojson, bma_districts_geojson = load_data()
except FileNotFoundError as e:
    st.error(
        f"Missing required data file: {e}. Please ensure `bangkok_master_cooling_dataset.csv` "
        "and `bangkok_wetland_boundary.geojson` are in the same repository folder as `app.py`."
    )
    st.stop()

# ---------------------------------------------------------
# 3. BANGKOK BOUNDARY, VEGETATION ZONES & COOLING CORRIDORS
# ---------------------------------------------------------
BANGKOK_BMA_BOUNDARY_COORDS = [
    [100.328, 13.785], [100.345, 13.865], [100.435, 13.925], [100.545, 13.955],
    [100.640, 13.945], [100.745, 13.925], [100.885, 13.895], [100.938, 13.835],
    [100.915, 13.735], [100.865, 13.685], [100.765, 13.675], [100.685, 13.655],
    [100.605, 13.665], [100.575, 13.685], [100.535, 13.675], [100.495, 13.645],
    [100.445, 13.595], [100.415, 13.525], [100.445, 13.495], [100.495, 13.515],
    [100.475, 13.575], [100.415, 13.625], [100.355, 13.665], [100.328, 13.785]
]

BANGKOK_WETLAND_SITES = [
    {
        "name": "Benjakitti Forest Park & Wetland (Calibration Site)",
        "district": "Khlong Toei / Asok CBD",
        "status": "Existing",
        "area_ha": 88.84,
        "use_geojson": True,
        "coords_wgs84": None,
    },
    {
        "name": "Lumphini Park & Retention Lake",
        "district": "Pathum Wan / Silom CBD",
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
        "district": "Ratchathewi / Rama IX CBD",
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
# 4. SIDEBAR: SIMULATION SCOPE, CBD MITIGATION PACKAGES & SLIDERS
# ---------------------------------------------------------
st.sidebar.header("🧭 1. Simulation Scale & Vision")

sim_mode = st.sidebar.radio(
    "Select Planning Scale:",
    options=[
        "Bangkok CBD Mitigation Lab (Benjakitti–Asok–Lumphini)",
        "Bangkok-Wide Vision (Boundary + Vegetation + Corridors)",
        "Benjakitti Empirical Site (22 Field Sensors Validation)"
    ],
    index=0
)

mitigation_preset = st.sidebar.selectbox(
    "Select Future Mitigation Package:",
    options=[
        "Baseline Existing CBD (2026 Current Condition)",
        "Moderate CBD Mitigation (30% Canopy + Wind Setbacks + Cool Roofs)",
        "Full Integrated CBD Vision (45% Canopy + Ventilation Corridors + Wetland Edge Opt.)",
        "Extreme Heatwave + Full CBD Mitigation (950 W/m² Stress Test)"
    ],
    index=2
)

if mitigation_preset == "Baseline Existing CBD (2026 Current Condition)":
    p_scale, p_canopy, p_openness, p_cool_surf, p_edge_opt, p_solar, p_wind = 100, 15, 40, 10, False, 650, 1.9
    include_future_nodes, corridor_boost_on = False, False
elif mitigation_preset == "Moderate CBD Mitigation (30% Canopy + Wind Setbacks + Cool Roofs)":
    p_scale, p_canopy, p_openness, p_cool_surf, p_edge_opt, p_solar, p_wind = 110, 30, 60, 40, True, 650, 2.0
    include_future_nodes, corridor_boost_on = True, True
elif mitigation_preset == "Full Integrated CBD Vision (45% Canopy + Ventilation Corridors + Wetland Edge Opt.)":
    p_scale, p_canopy, p_openness, p_cool_surf, p_edge_opt, p_solar, p_wind = 120, 45, 80, 70, True, 650, 2.1
    include_future_nodes, corridor_boost_on = True, True
else:
    p_scale, p_canopy, p_openness, p_cool_surf, p_edge_opt, p_solar, p_wind = 120, 45, 80, 70, True, 950, 1.2
    include_future_nodes, corridor_boost_on = True, True

st.sidebar.divider()
st.sidebar.header("🏗️ 2. CBD Urban Design Mitigation Levers")
st.sidebar.caption("Adjust built-environment & blue-green design parameters to test cooling gains.")

wetland_scale_pct = st.sidebar.slider(
    "1. Wetland Area Expansion / Link (%)",
    min_value=50,
    max_value=150,
    value=p_scale,
    step=10,
    help="Expanding connected wetland & retention water-forest area."
)

wetland_edge_opt = st.sidebar.checkbox(
    "2. Multi-Tier Riparian Canopy & Shallow Water Edge (+0.35°C)",
    value=p_edge_opt,
    help="Optimizes wetland edge evapotranspiration & shading."
)

cbd_canopy_pct = st.sidebar.slider(
    "3. CBD Street Tree Canopy & Green Roof Cover (%)",
    min_value=10,
    max_value=50,
    value=p_canopy,
    step=5,
    help="Bangkok CBD current baseline ~15%. Target 30–45% via street trees & sky gardens."
)

canyon_openness_pct = st.sidebar.slider(
    "4. Street Canyon Wind Porosity / Step-Down Massing (%)",
    min_value=30,
    max_value=90,
    value=p_openness,
    step=5,
    help="Aligned with 231° SW monsoon. Higher porosity reduces drag and extends cool air penetration."
)

cool_surface_pct = st.sidebar.slider(
    "5. High-Albedo Cool Roofs & Permeable Pavements (%)",
    min_value=10,
    max_value=80,
    value=p_cool_surf,
    step=10,
    help="Reduces urban heat storage in Asok, Sukhumvit, and Rama IV blocks."
)

st.sidebar.divider()
st.sidebar.header("🌦️ 3. Climate & Evaluation Setback")

target_distance_m = st.sidebar.slider(
    "Target CBD Street / Building Distance from Park (m)",
    min_value=0,
    max_value=800,
    value=200,
    step=25
)

wind_direction_deg = st.sidebar.slider(
    "Prevailing Wind Direction (° Azimuth)",
    min_value=0,
    max_value=360,
    value=231,
    step=5
)

wind_speed_ms = st.sidebar.slider(
    "Ambient Wind Speed (m/s)",
    min_value=0.5,
    max_value=5.0,
    value=p_wind,
    step=0.1
)

solar_rad_wm2 = st.sidebar.slider(
    "Peak Afternoon Solar Radiation (W/m²)",
    min_value=200,
    max_value=1000,
    value=p_solar,
    step=50
)

cooling_threshold_c = st.sidebar.selectbox(
    "Planning Cooling Threshold (°C)",
    options=[0.5, 1.0, 1.5],
    index=0
)

st.sidebar.divider()
st.sidebar.header("🗺️ 4. Map Layer Toggles")
show_bma_boundary = st.sidebar.checkbox("Show Bangkok Boundary (BMA)", value=True)
show_vegetation = st.sidebar.checkbox("Show Vegetation & Canopy Belts", value=True)
show_corridors = st.sidebar.checkbox("Show Blue-Green Cooling Corridors", value=True)

# ---------------------------------------------------------
# 5. ENHANCED EMPIRICAL + URBAN MITIGATION COOLING ENGINE
# ---------------------------------------------------------
def compute_mitigated_metrics(
    base_area_ha, scale_pct, solar_wm2, wind_ms, threshold_c, setback_m,
    canopy_pct=15, openness_pct=40, albedo_pct=10, edge_opt=False, corridor_active=False
):
    sim_area_ha = base_area_ha * (scale_pct / 100.0)
    area_factor = (sim_area_ha / 88.84) ** 0.35
    solar_adj = 0.0026 * (solar_wm2 - 650.0)
    wind_adj = -0.1884 * (wind_ms - 1.92)

    edge_boost = 0.35 if edge_opt else 0.0
    corridor_boost = 0.20 if corridor_active else 0.0
    canopy_lift = 0.015 * max(0, canopy_pct - 15)
    albedo_lift = 0.005 * max(0, albedo_pct - 10)
    local_cbd_lift = canopy_lift + albedo_lift

    openness_factor = 1.0 + 0.006 * max(0, openness_pct - 40)
    k = 0.00490 / (area_factor * openness_factor)
    core_plume_dt = max(0.2, (2.097 * area_factor) + solar_adj + wind_adj + edge_boost + corridor_boost)

    def pred(d, downwind=False):
        dw_mult = 1.18 if downwind else 1.0
        eff_d = d / dw_mult
        plume_dt = core_plume_dt * math.exp(-k * eff_d) - 0.050
        neighborhood_dt = local_cbd_lift * math.exp(-0.0012 * d)
        return max(0.0, round(plume_dt + neighborhood_dt, 2))

    mcd_cross = 0.0
    for step_d in range(0, 1501):
        if pred(step_d, False) >= threshold_c:
            mcd_cross = float(step_d)
        else:
            break

    mcd_down = 0.0
    for step_d in range(0, 1801):
        if pred(step_d, True) >= threshold_c:
            mcd_down = float(step_d)
        else:
            break

    return {
        "sim_area_ha": round(sim_area_ha, 1),
        "core_dt": round(core_plume_dt + local_cbd_lift, 2),
        "local_cbd_lift": round(local_cbd_lift, 2),
        "mcd_cross": mcd_cross,
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

def add_bangkok_planning_layers(fmap):
    if show_bma_boundary:
        if bma_districts_geojson is not None:
            folium.GeoJson(
                bma_districts_geojson,
                name="Bangkok BMA 50 Districts (OSM)",
                style_function=lambda x: {
                    "fillColor": "#f7fbff",
                    "color": "#d95f02",
                    "weight": 1.8,
                    "dashArray": "5, 4",
                    "fillOpacity": 0.03,
                }
            ).add_to(fmap)
        else:
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

    if show_vegetation:
        for vz in VEGETATION_ZONES:
            if "Future" in vz["type"] and not include_future_nodes:
                continue
            is_future = "Future" in vz["type"]
            v_poly = Polygon(vz["coords"])
            folium.GeoJson(
                mapping(v_poly),
                style_function=lambda x, fut=is_future: {
                    "fillColor": "#78c679" if not fut else "#addd8e",
                    "color": "#238443",
                    "weight": 1.2,
                    "dashArray": "4, 4" if fut else None,
                    "fillOpacity": 0.24 if not fut else 0.20,
                },
                tooltip=f"{vz['name']} ({vz['type']} | Canopy: {vz['canopy_pct']}%)"
            ).add_to(fmap)

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
                    "fillOpacity": 0.26,
                },
                tooltip=f"{corr['name']} ({corr['status']} | Extra Cooling: -{corr['bonus_dt']:.2f}°C)"
            ).add_to(fmap)

# =========================================================
# VIEW 1: BANGKOK CBD MITIGATION LAB (BENJAKITTI-ASOK-LUMPHINI)
# =========================================================
if sim_mode.startswith("Bangkok CBD"):
    base_res = compute_mitigated_metrics(88.84, 100, solar_rad_wm2, wind_speed_ms, cooling_threshold_c, target_distance_m, 15, 40, 10, False, False)
    mit_res = compute_mitigated_metrics(
        88.84, wetland_scale_pct, solar_rad_wm2, wind_speed_ms, cooling_threshold_c, target_distance_m,
        canopy_pct=cbd_canopy_pct, openness_pct=canyon_openness_pct, albedo_pct=cool_surface_pct,
        edge_opt=wetland_edge_opt, corridor_active=corridor_boost_on
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mitigated Core Cooling (ΔT)", f"-{mit_res['core_dt']:.2f} °C", f"-{mit_res['core_dt'] - base_res['core_dt']:.2f} °C Gain vs. Baseline")
    with col2:
        st.metric(f"CBD Cooling Reach (≥{cooling_threshold_c}°C)", f"{mit_res['mcd_cross']:.0f} m", f"+{mit_res['mcd_cross'] - base_res['mcd_cross']:.0f} m Extended Reach")
    with col3:
        st.metric("Downwind Corridor Reach", f"{mit_res['mcd_down']:.0f} m", f"+{mit_res['mcd_down'] - base_res['mcd_down']:.0f} m Along {wind_direction_deg}° Wind")
    with col4:
        st.metric(f"Cooling at {target_distance_m}m CBD Block", f"-{mit_res['dt_setback_down']:.2f} °C", f"-{mit_res['dt_setback_down'] - base_res['dt_setback_down']:.2f} °C Extra Relief")

    st.divider()
    map_col, chart_col = st.columns([1.3, 1.0])

    with map_col:
        st.subheader("🗺️ Central Bangkok CBD Cooling Expansion Map (Benjakitti – Asok – Lumphini)")
        poly_utm = transform(to_utm.transform, benjakitti_poly_wgs)
        if wetland_scale_pct != 100:
            s_lin = math.sqrt(wetland_scale_pct / 100.0)
            poly_utm_sim = scale(poly_utm, xfact=s_lin, yfact=s_lin, origin="center")
        else:
            poly_utm_sim = poly_utm

        downwind_rad = math.radians((wind_direction_deg + 180) % 360)

        base_outer = poly_utm.buffer(base_res["mcd_cross"])
        base_shift = (base_res["mcd_down"] - base_res["mcd_cross"]) * 0.8
        base_plume = translate(base_outer, xoff=base_shift * math.sin(downwind_rad), yoff=base_shift * math.cos(downwind_rad))

        mit_outer = poly_utm_sim.buffer(mit_res["mcd_cross"])
        mit_shift = (mit_res["mcd_down"] - mit_res["mcd_cross"]) * 0.8
        mit_plume = translate(mit_outer, xoff=mit_shift * math.sin(downwind_rad), yoff=mit_shift * math.cos(downwind_rad))

        m_cbd = folium.Map(location=[13.7335, 100.5535], zoom_start=14, tiles="OpenStreetMap", name="Bangkok Street Map")
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            name="Satellite Aerial View",
            overlay=False,
            control=True
        ).add_to(m_cbd)

        add_bangkok_planning_layers(m_cbd)

        folium.GeoJson(
            mapping(transform(to_wgs.transform, mit_plume)),
            name=f"Mitigated Future Cooling Zone (~{mit_res['mcd_cross']:.0f}m–{mit_res['mcd_down']:.0f}m)",
            style_function=lambda x: {"fillColor": "#02818a", "color": "#014636", "weight": 2.0, "fillOpacity": 0.24},
        ).add_to(m_cbd)

        folium.GeoJson(
            mapping(transform(to_wgs.transform, base_plume)),
            name=f"Existing Baseline Cooling Limit (~{base_res['mcd_cross']:.0f}m)",
            style_function=lambda x: {"fillColor": "none", "color": "#e31a1c", "weight": 2.2, "dashArray": "6, 4"},
        ).add_to(m_cbd)

        folium.GeoJson(
            mapping(transform(to_wgs.transform, poly_utm_sim)),
            name="Benjakitti Wetland Core",
            style_function=lambda x: {"fillColor": "#006d2c", "color": "#00441b", "weight": 2.5, "fillOpacity": 0.65},
        ).add_to(m_cbd)

        for site in BANGKOK_WETLAND_SITES[1:3]:
            sp_utm = transform(to_utm.transform, Polygon(site["coords_wgs84"]))
            sp_res = compute_mitigated_metrics(
                site["area_ha"], wetland_scale_pct, solar_rad_wm2, wind_speed_ms, cooling_threshold_c, target_distance_m,
                canopy_pct=cbd_canopy_pct, openness_pct=canyon_openness_pct, albedo_pct=cool_surface_pct,
                edge_opt=wetland_edge_opt, corridor_active=corridor_boost_on
            )
            sp_plume = translate(
                sp_utm.buffer(sp_res["mcd_cross"]),
                xoff=(sp_res["mcd_down"] - sp_res["mcd_cross"]) * 0.8 * math.sin(downwind_rad),
                yoff=(sp_res["mcd_down"] - sp_res["mcd_cross"]) * 0.8 * math.cos(downwind_rad)
            )
            folium.GeoJson(
                mapping(transform(to_wgs.transform, sp_plume)),
                style_function=lambda x: {"fillColor": "#3690c0", "color": "#034e7b", "weight": 1.2, "fillOpacity": 0.20},
            ).add_to(m_cbd)
            folium.GeoJson(
                mapping(transform(to_wgs.transform, sp_utm)),
                style_function=lambda x: {"fillColor": "#238b45", "color": "#00441b", "weight": 2.0, "fillOpacity": 0.65},
                tooltip=f"{site['name']} (Mitigated Core ΔT: -{sp_res['core_dt']:.2f}°C | Reach: {sp_res['mcd_down']:.0f}m)"
            ).add_to(m_cbd)

        folium.LayerControl(collapsed=True).add_to(m_cbd)
        st_folium(m_cbd, width=None, height=500, returned_objects=[])

    with chart_col:
        st.subheader("📈 Baseline vs. Mitigated CBD Cooling Curve")
        distances = np.linspace(0, 800, 160)
        curve_df = pd.DataFrame({
            "Distance from Wetland (m)": distances,
            "1. Existing Baseline (Downwind ΔT °C)": [base_res["predict_fn"](d, True) for d in distances],
            "2. Mitigated Future CBD (Downwind ΔT °C)": [mit_res["predict_fn"](d, True) for d in distances],
        }).set_index("Distance from Wetland (m)")
        st.line_chart(curve_df, height=300)

        msg = (
            f"**CBD Mitigation Gain Summary (at {target_distance_m}m into Asok / Sukhumvit / Rama IV):**\n\n"
            f"- **Without Mitigation (Baseline):** Peak cooling is **-{base_res['dt_setback_down']:.2f} °C** (reaches {base_res['mcd_down']:.0f} m).\n"
            f"- **With Selected CBD Mitigation:** Peak cooling increases to **-{mit_res['dt_setback_down']:.2f} °C** (**+{mit_res['dt_setback_down'] - base_res['dt_setback_down']:.2f} °C** extra cooling) and penetrates **{mit_res['mcd_down']:.0f} m** into dense CBD blocks."
        )
        st.success(msg)

    st.subheader("🔬 Factor-by-Factor Mitigation Contribution Breakdown (Thesis Planning Matrix)")
    factor_rows = [
        {
            "Mitigation Factor / Urban Design Intervention": "0. Baseline Observed (Current Benjakitti + Unmitigated CBD)",
            "Target Planning Parameter": "88.8 ha wetland, 15% CBD canopy, 40% porosity, 10% cool surfaces",
            "Peak Core ΔT (°C)": -base_res["core_dt"],
            "Downwind Reach ≥0.5°C (m)": base_res["mcd_down"],
            f"Cooling at {target_distance_m}m CBD Block (°C)": -base_res["dt_setback_down"],
            "Net Gain vs. Baseline (°C)": 0.00
        },
        {
            "Mitigation Factor / Urban Design Intervention": "+ Factor A: Wetland Riparian Canopy & Edge Optimization",
            "Target Planning Parameter": "Multi-tier shade trees + shallow littoral water edge (+0.35°C)",
            "Peak Core ΔT (°C)": -compute_mitigated_metrics(88.84, 100, solar_rad_wm2, wind_speed_ms, cooling_threshold_c, target_distance_m, 15, 40, 10, True, False)["core_dt"],
            "Downwind Reach ≥0.5°C (m)": compute_mitigated_metrics(88.84, 100, solar_rad_wm2, wind_speed_ms, cooling_threshold_c, target_distance_m, 15, 40, 10, True, False)["mcd_down"],
            f"Cooling at {target_distance_m}m CBD Block (°C)": -compute_mitigated_metrics(88.84, 100, solar_rad_wm2, wind_speed_ms, cooling_threshold_c, target_distance_m, 15, 40, 10, True, False)["dt_setback_down"],
            "Net Gain vs. Baseline (°C)": round(compute_mitigated_metrics(88.84, 100, solar_rad_wm2, wind_speed_ms, cooling_threshold_c, target_distance_m, 15, 40, 10, True, False)["dt_setback_down"] - base_res["dt_setback_down"], 2)
        },
        {
            "Mitigation Factor / Urban Design Intervention": "+ Factor B: CBD Street Tree Canopy & Green Roofs (45%)",
            "Target Planning Parameter": "Increase CBD green canopy from 15% to 45% in 0–400m buffer",
            "Peak Core ΔT (°C)": -compute_mitigated_metrics(88.84, 100, solar_rad_wm2, wind_speed_ms, cooling_threshold_c, target_distance_m, 45, 40, 10, False, False)["core_dt"],
            "Downwind Reach ≥0.5°C (m)": compute_mitigated_metrics(88.84, 100, solar_rad_wm2, wind_speed_ms, cooling_threshold_c, target_distance_m, 45, 40, 10, False, False)["mcd_down"],
            f"Cooling at {target_distance_m}m CBD Block (°C)": -compute_mitigated_metrics(88.84, 100, solar_rad_wm2, wind_speed_ms, cooling_threshold_c, target_distance_m, 45, 40, 10, False, False)["dt_setback_down"],
            "Net Gain vs. Baseline (°C)": round(compute_mitigated_metrics(88.84, 100, solar_rad_wm2, wind_speed_ms, cooling_threshold_c, target_distance_m, 45, 40, 10, False, False)["dt_setback_down"] - base_res["dt_setback_down"], 2)
        },
        {
            "Mitigation Factor / Urban Design Intervention": "+ Factor C: Ventilation Corridor & Step-Down Massing (80% Porosity)",
            "Target Planning Parameter": "Align street canyons & height setbacks with 231° SW monsoon",
            "Peak Core ΔT (°C)": -compute_mitigated_metrics(88.84, 100, solar_rad_wm2, wind_speed_ms, cooling_threshold_c, target_distance_m, 15, 80, 10, False, True)["core_dt"],
            "Downwind Reach ≥0.5°C (m)": compute_mitigated_metrics(88.84, 100, solar_rad_wm2, wind_speed_ms, cooling_threshold_c, target_distance_m, 15, 80, 10, False, True)["mcd_down"],
            f"Cooling at {target_distance_m}m CBD Block (°C)": -compute_mitigated_metrics(88.84, 100, solar_rad_wm2, wind_speed_ms, cooling_threshold_c, target_distance_m, 15, 80, 10, False, True)["dt_setback_down"],
            "Net Gain vs. Baseline (°C)": round(compute_mitigated_metrics(88.84, 100, solar_rad_wm2, wind_speed_ms, cooling_threshold_c, target_distance_m, 15, 80, 10, False, True)["dt_setback_down"] - base_res["dt_setback_down"], 2)
        },
        {
            "Mitigation Factor / Urban Design Intervention": "+ Factor D: High-Albedo Cool Roofs & Pavements (70%)",
            "Target Planning Parameter": "Reflective roof coatings & permeable cool asphalt in CBD",
            "Peak Core ΔT (°C)": -compute_mitigated_metrics(88.84, 100, solar_rad_wm2, wind_speed_ms, cooling_threshold_c, target_distance_m, 15, 40, 70, False, False)["core_dt"],
            "Downwind Reach ≥0.5°C (m)": compute_mitigated_metrics(88.84, 100, solar_rad_wm2, wind_speed_ms, cooling_threshold_c, target_distance_m, 15, 40, 70, False, False)["mcd_down"],
            f"Cooling at {target_distance_m}m CBD Block (°C)": -compute_mitigated_metrics(88.84, 100, solar_rad_wm2, wind_speed_ms, cooling_threshold_c, target_distance_m, 15, 40, 70, False, False)["dt_setback_down"],
            "Net Gain vs. Baseline (°C)": round(compute_mitigated_metrics(88.84, 100, solar_rad_wm2, wind_speed_ms, cooling_threshold_c, target_distance_m, 15, 40, 70, False, False)["dt_setback_down"] - base_res["dt_setback_down"], 2)
        },
        {
            "Mitigation Factor / Urban Design Intervention": "★ COMBINED FULL CBD MITIGATION (All Factors Active)",
            "Target Planning Parameter": "Integrated Wetland + 45% Canopy + Wind Corridor + 70% Cool Surfaces",
            "Peak Core ΔT (°C)": -mit_res["core_dt"],
            "Downwind Reach ≥0.5°C (m)": mit_res["mcd_down"],
            f"Cooling at {target_distance_m}m CBD Block (°C)": -mit_res["dt_setback_down"],
            "Net Gain vs. Baseline (°C)": round(mit_res["dt_setback_down"] - base_res["dt_setback_down"], 2)
        }
    ]
    factor_df = pd.DataFrame(factor_rows)
    st.dataframe(factor_df, use_container_width=True, hide_index=True)

    st.download_button(
        label="⬇️ Download CBD Mitigation Factor Table (CSV)",
        data=factor_df.to_csv(index=False).encode("utf-8"),
        file_name="bangkok_cbd_mitigation_performance.csv",
        mime="text/csv"
    )

# =========================================================
# VIEW 2: BANGKOK-WIDE VISION (EXISTING VS FUTURE SCENARIO)
# =========================================================
elif sim_mode.startswith("Bangkok-Wide"):
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
        s_res = compute_mitigated_metrics(
            site["area_ha"], wetland_scale_pct, solar_rad_wm2, wind_speed_ms,
            cooling_threshold_c, target_distance_m,
            canopy_pct=cbd_canopy_pct, openness_pct=canyon_openness_pct, albedo_pct=cool_surface_pct,
            edge_opt=wetland_edge_opt, corridor_active=corridor_boost_on
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

    all_cooled_geoms = list(plume_geoms_utm)
    if show_corridors:
        for corr in COOLING_CORRIDORS:
            if "Future" in corr["status"] and not corridor_boost_on:
                continue
            c_utm = transform(to_utm.transform, LineString(corr["coords"])).buffer(corr["width_m"])
            all_cooled_geoms.append(c_utm)

    merged_core_ha = unary_union(core_geoms_utm).area / 10000.0
    merged_cooled_ha = max(0.0, (unary_union(all_cooled_geoms).area / 10000.0) - merged_core_ha)

    active_veg = [v for v in VEGETATION_ZONES if "Future" not in v["type"] or include_future_nodes]
    veg_geoms_utm = [transform(to_utm.transform, Polygon(v["coords"])) for v in active_veg]
    merged_veg_ha = unary_union(veg_geoms_utm).area / 10000.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Wetland Core Area", f"{merged_core_ha:.1f} ha", f"{merged_core_ha - 798.4:+.1f} ha vs. Existing")
    with col2:
        st.metric("Urban Vegetation & Canopy Zones", f"{merged_veg_ha:,.0f} ha", f"{len(active_veg)} Green Belts Mapped")
    with col3:
        st.metric("Net Cooled Urban Zone (≥0.5°C)", f"{merged_cooled_ha:,.1f} ha", f"{merged_cooled_ha - 1185.0:+.1f} ha vs. Existing")
    with col4:
        avg_core_dt = np.mean([r["Peak Core ΔT (°C)"] for r in city_rows])
        st.metric("Network Mean Core Cooling", f"{avg_core_dt:.2f} °C", "Mitigation Active" if corridor_boost_on else "Baseline")

    st.divider()
    folium.LayerControl(collapsed=True).add_to(m_city)
    st_folium(m_city, width=None, height=540, returned_objects=[])

    st.subheader("📋 Node-by-Node Bangkok Cooling Performance Table")
    city_df = pd.DataFrame(city_rows)
    st.dataframe(city_df, use_container_width=True, hide_index=True)

    st.download_button(
        label="⬇️ Download Bangkok Scenario Results (CSV)",
        data=city_df.to_csv(index=False).encode("utf-8"),
        file_name="bangkok_citywide_mitigation_scenarios.csv",
        mime="text/csv"
    )

# =========================================================
# VIEW 3: BENJAKITTI EMPIRICAL SITE (22 SENSORS VALIDATION)
# =========================================================
else:
    res = compute_mitigated_metrics(
        88.84, wetland_scale_pct, solar_rad_wm2, wind_speed_ms,
        cooling_threshold_c, target_distance_m,
        canopy_pct=cbd_canopy_pct, openness_pct=canyon_openness_pct, albedo_pct=cool_surface_pct,
        edge_opt=wetland_edge_opt, corridor_active=corridor_boost_on
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
        st.subheader("🗺️ Benjakitti Empirical Site & 22 Outdoor Loggers")
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

        site_msg = (
            f"**Site Planning Insight (at {target_distance_m}m Setback):**\n\n"
            f"- **Crosswind / General Street:** Expected peak afternoon cooling is **{res['dt_setback_cross']:.2f} °C**.\n"
            f"- **Downwind Ventilation Corridor ({wind_direction_deg}°):** Expected peak cooling reaches **{res['dt_setback_down']:.2f} °C**.\n"
            f"- **Zoning Guideline:** Preserve open street canyons and low-rise step-downs within **{res['mcd_cross']:.0f} m – {res['mcd_down']:.0f} m** of the wetland boundary."
        )
        st.info(site_msg)

    with st.expander("📊 View Field Sensor Summary Table (22 Validated Outdoor Loggers)"):
        st.dataframe(
            summary_df[[
                "sensor_id", "zone_type", "distance_to_wetland_m", "wind_exposure",
                "mean_T_24h", "peak_T_afternoon", "peak_DeltaT_afternoon", "peak_HI_afternoon"
            ]].sort_values("distance_to_wetland_m"),
            use_container_width=True,
            hide_index=True
        )
