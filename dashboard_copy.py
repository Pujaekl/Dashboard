import json
import re
from collections import Counter
import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import gaussian_kde
from matplotlib.backends.backend_pdf import PdfPages
import tempfile
from folium.plugins import Fullscreen, MiniMap
from streamlit_folium import st_folium

st.set_page_config(page_title="Vehicle Path Analytics Dashboard",page_icon="🚜",layout="wide",initial_sidebar_state="expanded")
COLORS = {
    "orange": "#E85D04",
    "orange_dark": "#9A3412",
    "charcoal": "#1F2933",
    "steel": "#52606D",
    "amber": "#FFB100",
    "amber_dark": "#B45309",
    "ivory": "#F7F4EF",
    "card": "#FFFFFF",
    "border": "#E2DDD3",
    "success": "#2E7D32",
    "muted": "#7B8794",
}

STAGE_COLORS = {
    "Overall":  {"cte": COLORS["steel"],  "heading": COLORS["charcoal"]},
    "Filtered": {"cte": COLORS["amber"],  "heading": COLORS["amber_dark"]},
    "Sliced":   {"cte": COLORS["orange"], "heading": COLORS["orange_dark"]},
}

plt.rcParams.update({
    "figure.facecolor": COLORS["card"],
    "axes.facecolor": COLORS["card"],
    "axes.edgecolor": COLORS["border"],
    "axes.labelcolor": COLORS["charcoal"],
    "text.color": COLORS["charcoal"],
    "xtick.color": COLORS["charcoal"],
    "ytick.color": COLORS["charcoal"],
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "grid.color": COLORS["border"],
    "font.size": 10,
})

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Big+Shoulders+Stencil+Display:wght@700;800&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', -apple-system, sans-serif; }
.main { background-color: #F7F4EF; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }

h1, h2, h3 { color: #1F2933; }
h2, h3 { border-left: 4px solid #E85D04; padding-left: 0.6rem; }

/* ---- Header nameplate ---- */
.ekl-plate {
    background:
        repeating-linear-gradient(0deg, rgba(255,255,255,0.04) 0px, rgba(255,255,255,0.04) 1px, transparent 1px, transparent 26px),
        repeating-linear-gradient(90deg, rgba(255,255,255,0.04) 0px, rgba(255,255,255,0.04) 1px, transparent 1px, transparent 26px),
        linear-gradient(120deg, #1F2933 0%, #2D3A45 55%, #1F2933 100%);
    border-left: 7px solid #E85D04;
    border-radius: 10px;
    padding: 1.3rem 1.7rem;
    margin-bottom: 1.3rem;
    box-shadow: 0 4px 14px rgba(0,0,0,0.18);
}
.ekl-plate h1 {
    font-family: 'Big Shoulders Stencil Display', 'Inter', sans-serif;
    color: #FFFFFF;
    font-size: 2.3rem;
    letter-spacing: 1.5px;
    margin: 0 0 0.3rem 0;
    text-transform: uppercase;
}
.ekl-plate p {
    font-family: 'JetBrains Mono', monospace;
    color: #FFB100;
    font-size: 0.78rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 0;
}

/* ---- KPI metric cards ---- */
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 27px;
    color: #1F2933;
    font-weight: 700;
}
[data-testid="stMetricLabel"] {
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 12px;
    color: #7B8794;
}
div[data-testid="metric-container"], div[data-testid="stMetric"] {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 16px 16px 14px 16px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
    border-top: 4px solid transparent;
    border-image: linear-gradient(90deg, #52606D, #FFB100, #E85D04) 1;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
div[data-testid="metric-container"]:hover, div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0px 6px 16px rgba(0,0,0,0.12);
}

/* ---- Tabs ---- */
[data-testid="stTabs"] button {
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-size: 13px;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #E85D04 !important;
    border-bottom-color: #E85D04 !important;
}

/* ---- Dividers ---- */
hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, #E2DDD3 20%, #E2DDD3 80%, transparent);
}

/* ---- Buttons ---- */
[data-testid="stDownloadButton"] button, .stButton button {
    background: #E85D04;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: background 0.15s ease, transform 0.15s ease;
}
[data-testid="stDownloadButton"] button:hover, .stButton button:hover {
    background: #9A3412;
    transform: translateY(-1px);
    color: white;
}

/* ---- Alerts & dataframes ---- */
[data-testid="stAlert"] { border-radius: 10px; }
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; border: 1px solid #E2DDD3; }

/* ---- Sidebar control badge ---- */
.ekl-sidebar-badge {
    background: #1F2933;
    border-left: 5px solid #E85D04;
    border-radius: 8px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 1rem;
}
.ekl-sidebar-badge p {
    font-family: 'JetBrains Mono', monospace;
    color: #FFB100;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-size: 0.72rem;
    margin: 0;
}
</style>

<div class="ekl-plate">
  <h1>🚜 Vehicle Path Analytics</h1>
  <p>Overall &nbsp;&middot;&nbsp; Filtered &nbsp;&middot;&nbsp; Sliced &nbsp;&middot;&nbsp; CTE &nbsp;&middot;&nbsp; Heading Error</p>
</div>
""", unsafe_allow_html=True)
@st.cache_data
def load_and_preprocess_log_data(uploaded_file):
    content = uploaded_file.read().decode("utf-8",errors="ignore")
    matches = re.findall(r'\{.*?\}',content,re.DOTALL)
    data = []
    for m in matches:
        try:
            data.append(json.loads(m))
        except:
            pass
    total_raw_records = len(data)
    df = pd.DataFrame(data)
    numeric_cols = ['Latitude_A','Longitude_A','Latitude_B','Longitude_B','Latitude','Longitude','CTE','Heading error','velocity','Heading']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col],errors='coerce')
    df_clean = df.dropna(subset=['Latitude', 'Longitude'])

    return df_clean, total_raw_records
    
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat / 2) ** 2+ np.cos(lat1)* np.cos(lat2)* np.sin(dlon / 2) ** 2)
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c * 1000

def remove_coordinate_outliers(df, columns=['Latitude', 'Longitude'], method='iqr', factor=1.5):
    """
    Identifies and removes coordinate outliers from the dataframe.
    Methods: 'iqr' (Interquartile Range) or 'z_score' (Standard Deviation)
    """
    if df.empty:
        return df       
    mask = pd.Series(True, index=df.index)
    
    for col in columns:
        if col in df.columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                mask &= (df[col] >= lower_bound) & (df[col] <= upper_bound)
                         
    return df[mask].reset_index(drop=True)

def calculate_dynamic_distance_threshold(path_points,min_base_threshold=10):
    """
    Calculates dynamic threshold using minimum
    significant segment distance."""
    
    distances = []
    if path_points and len(path_points) > 1:
        for i in range(len(path_points) - 1):
            p1 = path_points[i]
            p2 = path_points[i + 1]
            dist = haversine(p1[0],p1[1],p2[0],p2[1] )
            if dist > min_base_threshold:
                distances.append(dist)
    if distances:
        return min(distances)
    return min_base_threshold

def analyze_checkpoints(df_clean,distance_threshold_m=35):
    filtered_checkpoints = {}
    checkpoint_summary = []
    for cp, group in df_clean.groupby('checkpoint'):
        if cp == 0:
            continue
        path_points = list(zip(group['Latitude'],group['Longitude']))

# CALCULATE DYNAMIC THRESHOLD
        dynamic_threshold = calculate_dynamic_distance_threshold(path_points,min_base_threshold=distance_threshold_m)

# UNIQUE SEGMENTS
        points_list = list(zip(group['Latitude_A'],group['Longitude_A'],group['Latitude_B'],group['Longitude_B'],group['CTE'],group['Heading error']))
        counts = Counter(points_list)
        valid_trips = []
        total_segments = 0
        straight_segments = 0
# DISTANCE FILTERING
        for p, count in counts.items():
            total_segments += 1
            dist_m = haversine(p[0], p[1],p[2],p[3])
# STRAIGHT PATH FILTER
            if dist_m >= dynamic_threshold:
                straight_segments += 1
                valid_trips.append({
                    'points': [p[0],p[1], p[2],p[3]],
                    'distance_m': dist_m,
                    'count': count,
                    'cte': p[4],
                    'heading_error': p[5],
                    'dynamic_threshold': dynamic_threshold})

# STORE FILTERED CHECKPOINTS
        if valid_trips:
            filtered_checkpoints[cp] = valid_trips
        checkpoint_summary.append({"Checkpoint": cp,"Dynamic Threshold (m)": round(dynamic_threshold,2),
            "Total Segments": total_segments,
            "Straight Segments": straight_segments,
            "Filtered Ratio (%)": round(( straight_segments / total_segments) * 100,2) if total_segments > 0 else 0})

    all_ctes = []
    all_headings = []
    for cp, trips in filtered_checkpoints.items():
        for trip in trips:
            all_ctes.extend([trip['cte']] * trip['count'])
            all_headings.extend([trip['heading_error']] * trip['count'])
    checkpoint_summary_df = pd.DataFrame(checkpoint_summary)

    return (filtered_checkpoints,all_ctes,all_headings,checkpoint_summary_df)

def calculate_metrics(cte_values,heading_values,threshold=1):
    mean_abs_cte = stats.mode(np.abs(cte_values), keepdims=True).mode[0]
    mean_abs_heading = stats.mode(np.abs(heading_values), keepdims=True).mode[0]
    cte_conf = (len([x for x in cte_values if abs(x) <= threshold])/ len(cte_values)) * 100 if cte_values else 0
    heading_conf = (len([x for x in heading_values if abs(x) <= threshold])/ len(heading_values)) * 100 if heading_values else 0
    return (mean_abs_cte,mean_abs_heading,cte_conf,heading_conf)

def get_overall_sin_theta_metric(mean_error_deg):
    """
    Calculates the sine of the absolute mean heading error
    and scales it by a factor of 2.26.
    """
    error_radians = np.radians(np.abs(mean_error_deg))
    metric_value = (np.sin(error_radians) * 2.26)
    return metric_value
    
def get_sliced_sin_theta_metric(mean_error_deg_sliced):
    """
    Calculates the sine of the absolute mean heading error (sliced)
    and scales it by a factor of 2.26."""
    error_radians = np.radians(np.abs(mean_error_deg_sliced))
    metric_value = (np.sin(error_radians) * 2.26)
    return metric_value

def plot_distribution(data,title,color):
    fig, ax = plt.subplots(figsize=(8, 4))
    abs_data = np.abs(data)
    sns.kdeplot(abs_data,fill=True,color=color,bw_adjust=0.5,ax=ax)
    mode_result = stats.mode(np.round(abs_data, 3), keepdims=True)
    mode_val = float(mode_result.mode[0])
    ax.axvline(mode_val,color='red',linestyle='--',linewidth=2,label=f"{mode_val:.2f}")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    return fig

def plot_mean_metrics_bar(overall_cte,filtered_cte,sliced_cte,overall_heading,filtered_heading,sliced_heading):
    fig, axes = plt.subplots(1, 2,figsize=(12, 5))
    cte_labels = ['Overall','Filtered','Sliced']
    cte_values = [overall_cte,filtered_cte,sliced_cte]
    cte_colors = [STAGE_COLORS[s]["cte"] for s in cte_labels]
    heading_colors = [STAGE_COLORS[s]["heading"] for s in cte_labels]
    axes[0].bar(cte_labels,cte_values,color=cte_colors)
    axes[0].set_title('Mean Absolute CTE')
    axes[0].set_ylabel('Meters')
    axes[0].grid(alpha=0.3)
    for i, v in enumerate(cte_values):
        axes[0].text(i,v + 0.01,f"{v:.2f}",ha='center',fontweight='bold')

    heading_labels = ['Overall','Filtered','Sliced']
    heading_values = [overall_heading,filtered_heading,sliced_heading]
    axes[1].bar(heading_labels,heading_values,color=heading_colors)
    axes[1].set_title('Mean Absolute Heading Error')
    axes[1].set_ylabel('Degrees')
    axes[1].grid(alpha=0.3)
    for i, v in enumerate(heading_values):
        axes[1].text(i,v + 0.01,f"{v:.2f}",ha='center',fontweight='bold')
    plt.tight_layout()
    return fig

def get_filtered_path_data(df_clean,filtered_checkpoints):
    unique_segments = set()
    for cp, trips in filtered_checkpoints.items():
        for trip in trips:
            unique_segments.add(tuple(trip['points']))
    df_temp = df_clean.copy()
    df_temp['segment_id'] = list(zip(df_temp['Latitude_A'],df_temp['Longitude_A'],df_temp['Latitude_B'],df_temp['Longitude_B']))

    filtered_df = df_temp[df_temp['segment_id'].isin(unique_segments)]
    return filtered_df
    
def get_sliced_path_data(filtered_df,filtered_checkpoints):
    sorted_cps = sorted(filtered_checkpoints.keys())
    sliced_cps = (
        sorted_cps[1:10]
        if len(sorted_cps) > 4
        else sorted_cps)

    sliced_df = filtered_df[
        filtered_df['checkpoint'].isin(
            sliced_cps )]
    return sliced_df

def plot_vehicle_path_matplotlib(df, title, color, segmented=False):
    fig, ax = plt.subplots(figsize=(10, 10))
    if segmented and 'checkpoint' in df.columns:
        for cp, group in df.groupby('checkpoint'):
            ax.plot(group['Longitude'], group['Latitude'],
                    color=color, linewidth=2, marker='o', markersize=2, alpha=0.8)
    else:
        ax.plot(df['Longitude'], df['Latitude'],
                color=color, linewidth=2, marker='o', markersize=2, alpha=0.8)
                
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.3)
    ax.axis('equal') 
    return fig


# FOLIUM MAP
def create_folium_map(df,title,color,segmented=False):
    mean_lat = df['Latitude'].mean()
    mean_lon = df['Longitude'].mean()
    m = folium.Map(
        location=[mean_lat, mean_lon],
        zoom_start=40,
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        control_scale=True)

    Fullscreen().add_to(m)
    MiniMap().add_to(m)
  
# SEGMENTED PATH   
    if segmented and 'checkpoint' in df.columns:
        for cp, group in df.groupby('checkpoint'):
            coords = group[
                ['Latitude', 'Longitude']
            ].dropna().values.tolist()

            folium.PolyLine(
                coords,
                color=color,
                weight=5,
                opacity=0.9,
                tooltip=f"{title} - CP {cp}"
            ).add_to(m)
    else:
        coords = df[
            ['Latitude', 'Longitude']
        ].dropna().values.tolist()

        folium.PolyLine(
            coords,
            color=color,
            weight=5,
            opacity=0.9,
            tooltip=title
        ).add_to(m)    
    return m

# SIDEBAR
st.sidebar.markdown(
    '<div class="ekl-sidebar-badge"><p>⚙️ Dashboard Controls</p></div>',
    unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader(
    "Upload Vehicle Log File",
    type=["json", "txt"])

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.1,
    5.0,
    1.0)
   
def compute_cte_confidence_at_thresholds(cte_values, thresholds):
    cte_values = np.array(cte_values)
    results = []

    for t in thresholds:
        conf = (np.sum(np.abs(cte_values) <= t) / len(cte_values) * 100) if len(cte_values) else 0
        results.append({"Threshold (m)": t, "Confidence (%)": round(conf, 2)})
    return pd.DataFrame(results)

def compute_cte_for_confidence(cte_values, levels):
    cte_values = np.sort(np.abs(cte_values))
    n = len(cte_values)
    results = []
    for c in levels:
        if n == 0:
            results.append({
                "Confidence (%)": c,
                "CTE Required (m)": float("nan"),
                "CTE (cm)": float("nan")})
            continue
        idx = int(np.ceil((c/100)*n)) - 1
        idx = max(0, min(idx, n-1))

        results.append({
            "Confidence (%)": c,
            "CTE Required (m)": round(cte_values[idx], 5),
            "CTE (cm)": round(cte_values[idx]*100, 2)})
    return pd.DataFrame(results)

def run_cte_analysis(cte_values):
    thresholds = [0.05,0.10,0.20,0.30,0.40,0.50,1.0,1.2,1.5]
    levels = [60,80,90,95,100]
    return (
        compute_cte_confidence_at_thresholds(cte_values, thresholds),
        compute_cte_for_confidence(cte_values, levels))


PLOTLY_FONT = dict(family="Inter, -apple-system, sans-serif", color=COLORS["charcoal"])

def _hex_to_rgba(hex_color, alpha=0.18):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def _kde_curve(data, bw_adjust=0.5, n_points=200):
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) < 2 or np.isclose(data.std(), 0):
        return None, None
    kde = gaussian_kde(data)
    kde.set_bandwidth(kde.factor * bw_adjust)
    pad = (data.max() - data.min()) * 0.15 or 1.0
    xs = np.linspace(data.min() - pad, data.max() + pad, n_points)
    return xs, kde(xs)

def plot_distribution_plotly(data, title, color):
    abs_data = np.abs(np.asarray(data, dtype=float))
    abs_data = abs_data[~np.isnan(abs_data)]
    fig = go.Figure()
    xs, ys = _kde_curve(abs_data)
    if xs is not None:
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", line=dict(color=color, width=2.5),
            fill="tozeroy", fillcolor=_hex_to_rgba(color, 0.18),
            hovertemplate="Value: %{x:.3f}<br>Density: %{y:.3f}<extra></extra>", showlegend=False))
    # Show mode (peak of KDE) instead of mean
    if xs is not None and ys is not None:
        mode_val = float(xs[np.argmax(ys)])
    elif len(abs_data):
        mode_result = stats.mode(np.round(abs_data, 3), keepdims=True)
        mode_val = float(mode_result.mode[0])
    else:
        mode_val = 0.0
    fig.add_vline(x=mode_val, line_dash="dash", line_color=COLORS["orange_dark"], line_width=2,
                  annotation_text=f"{mode_val:.2f}", annotation_font_color=COLORS["charcoal"])
    fig.update_layout(title=title, height=340, margin=dict(l=40, r=20, t=50, b=40),
                       plot_bgcolor=COLORS["card"], paper_bgcolor=COLORS["card"], font=PLOTLY_FONT)
    fig.update_xaxes(gridcolor=COLORS["border"], zeroline=False, tickfont=dict(color=COLORS["charcoal"]))
    fig.update_yaxes(title="Density", gridcolor=COLORS["border"], zeroline=False, tickfont=dict(color=COLORS["charcoal"]))
    return fig

def plot_mean_metrics_bar_plotly(overall_cte, filtered_cte, sliced_cte, overall_heading, filtered_heading, sliced_heading):
    stages = ["Overall", "Filtered", "Sliced"]
    cte_values = [overall_cte, filtered_cte, sliced_cte]
    heading_values = [overall_heading, filtered_heading, sliced_heading]
    cte_colors = [STAGE_COLORS[s]["cte"] for s in stages]
    heading_colors = [STAGE_COLORS[s]["heading"] for s in stages]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Mean Absolute CTE (m)", "Mean Absolute Heading Error (deg)"))
    fig.add_trace(go.Bar(x=stages, y=cte_values, marker_color=cte_colors,
                          text=[f"{v:.2f}" for v in cte_values], textposition="outside",
                          hovertemplate="%{x}: %{y:.3f} m<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=stages, y=heading_values, marker_color=heading_colors,
                          text=[f"{v:.2f}" for v in heading_values], textposition="outside",
                          hovertemplate="%{x}: %{y:.3f}°<extra></extra>"), row=1, col=2)
    fig.update_layout(height=380, showlegend=False, margin=dict(l=40, r=20, t=60, b=40),
                       plot_bgcolor=COLORS["card"], paper_bgcolor=COLORS["card"], font=PLOTLY_FONT)
    fig.update_yaxes(gridcolor=COLORS["border"], zeroline=False, tickfont=dict(color=COLORS["charcoal"]))
    fig.update_xaxes(gridcolor=COLORS["border"], tickfont=dict(color=COLORS["charcoal"]))
    return fig

def plot_vehicle_path_plotly(df, title, color, segmented=False):
    fig = go.Figure()
    if segmented and "checkpoint" in df.columns:
        for cp, group in df.groupby("checkpoint"):
            fig.add_trace(go.Scattergl(
                x=group["Longitude"], y=group["Latitude"], mode="lines+markers",
                line=dict(color=color, width=2), marker=dict(size=3, color=color),
                name=f"CP {cp}", hovertemplate="Lon: %{x:.6f}<br>Lat: %{y:.6f}<extra></extra>"))
    else:
        fig.add_trace(go.Scattergl(
            x=df["Longitude"], y=df["Latitude"], mode="lines+markers",
            line=dict(color=color, width=2), marker=dict(size=3, color=color),
            name=title, hovertemplate="Lon: %{x:.6f}<br>Lat: %{y:.6f}<extra></extra>"))
    fig.update_layout(title=title, height=500, margin=dict(l=40, r=20, t=50, b=40),
                       plot_bgcolor=COLORS["card"], paper_bgcolor=COLORS["card"], font=PLOTLY_FONT,
                       showlegend=segmented)
    fig.update_xaxes(title="Longitude", gridcolor=COLORS["border"], zeroline=False, scaleanchor="y", scaleratio=1, tickfont=dict(color=COLORS["charcoal"]))
    fig.update_yaxes(title="Latitude", gridcolor=COLORS["border"], zeroline=False, tickfont=dict(color=COLORS["charcoal"]))
    return fig

def plot_cte_confidence_curve_plotly(overall_df, filtered_df, sliced_df, overall_cte, filtered_cte, sliced_cte):
    fig = go.Figure()
    for stage, df_s, cte_vals in [("Overall", overall_df, overall_cte), ("Filtered", filtered_df, filtered_cte), ("Sliced", sliced_df, sliced_cte)]:
        color = STAGE_COLORS[stage]["cte"]
        fig.add_trace(go.Scatter(x=df_s["Threshold (m)"], y=df_s["Confidence (%)"], mode="lines+markers",
                                  line=dict(color=color, width=2.5), marker=dict(size=6, color=color),
                                  name=stage, hovertemplate=stage + ": %{y:.1f}%% @ %{x} m<extra></extra>"))
        data = np.abs(np.asarray(cte_vals, dtype=float))
        data = data[~np.isnan(data)]
        if len(data):
            mu, sigma = float(np.mean(data)), float(np.std(data))
            fig.add_vline(x=mu, line_dash="dash", line_color=color, opacity=0.8)
            fig.add_vrect(x0=mu - sigma, x1=mu + sigma, fillcolor=color, opacity=0.08, line_width=0)
            fig.add_annotation(x=mu, y=8, text=f"{stage}<br>\u03bc={mu:.2f} \u03c3={sigma:.2f}",
                                showarrow=False, font=dict(color=color, size=10), textangle=-90)
    fig.update_layout(title="CTE Threshold vs Confidence", height=420, margin=dict(l=40, r=20, t=60, b=40),
                       plot_bgcolor=COLORS["card"], paper_bgcolor=COLORS["card"], font=PLOTLY_FONT,
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(title="CTE Threshold (m)", gridcolor=COLORS["border"], zeroline=False, tickfont=dict(color=COLORS["charcoal"]))
    fig.update_yaxes(title="Confidence (%)", gridcolor=COLORS["border"], zeroline=False, tickfont=dict(color=COLORS["charcoal"]))
    return fig

def plot_required_cte_curve_plotly(overall_df, filtered_df, sliced_df):
    fig = go.Figure()
    for stage, df_s in [("Overall", overall_df), ("Filtered", filtered_df), ("Sliced", sliced_df)]:
        color = STAGE_COLORS[stage]["cte"]
        fig.add_trace(go.Scatter(x=df_s["Confidence (%)"], y=df_s["CTE Required (m)"], mode="lines+markers",
                                  line=dict(color=color, width=2.5), marker=dict(size=6, color=color),
                                  name=stage, hovertemplate=stage + ": %{y:.4f} m @ %{x}%%<extra></extra>"))
    fig.update_layout(title="Confidence vs Required CTE", height=420, margin=dict(l=40, r=20, t=60, b=40),
                       plot_bgcolor=COLORS["card"], paper_bgcolor=COLORS["card"], font=PLOTLY_FONT,
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(title="Confidence (%)", gridcolor=COLORS["border"], zeroline=False, tickfont=dict(color=COLORS["charcoal"]))
    fig.update_yaxes(title="Required CTE (m)", gridcolor=COLORS["border"], zeroline=False, tickfont=dict(color=COLORS["charcoal"]))
    return fig


def plot_velocity_chart_plotly(df, step=50):
    if 'velocity' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No 'velocity' field found in log data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=14))
        fig.update_layout(title="Raw Velocity", height=360,
                           plot_bgcolor=COLORS["card"], paper_bgcolor=COLORS["card"], font=PLOTLY_FONT)
        return fig
    df_v = df.dropna(subset=['velocity']).reset_index(drop=True)
    if df_v.empty:
        fig = go.Figure()
        fig.add_annotation(text="No velocity data available", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=14))
        fig.update_layout(title="Raw Velocity", height=360,
                           plot_bgcolor=COLORS["card"], paper_bgcolor=COLORS["card"], font=PLOTLY_FONT)
        return fig
    df_sampled = df_v.iloc[::step].reset_index(drop=True)
    vel = df_sampled['velocity'].values
    mean_vel = float(np.mean(df_v['velocity']))  
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_sampled.index.tolist(), y=vel.tolist(), mode="lines",
        line=dict(color=COLORS["orange"], width=1.5),
        fill="tozeroy", fillcolor=_hex_to_rgba(COLORS["orange"], 0.12),
        name="Velocity", hovertemplate="Sample %{x}: %{y:.4f}<extra></extra>"))
    fig.add_hline(y=mean_vel, line_dash="dash", line_color=COLORS["amber_dark"], line_width=2,
                  annotation_text=f"Mean = {mean_vel:.4f}", annotation_font_color=COLORS["charcoal"])
    fig.update_layout(
        title=f"Raw Velocity ",
        height=360, margin=dict(l=40, r=20, t=55, b=40),
        plot_bgcolor=COLORS["card"], paper_bgcolor=COLORS["card"], font=PLOTLY_FONT,
        showlegend=False)
    fig.update_xaxes(title=f" Data ", gridcolor=COLORS["border"], zeroline=False, tickfont=dict(color=COLORS["charcoal"]))
    fig.update_yaxes(title="Velocity", gridcolor=COLORS["border"], zeroline=False, tickfont=dict(color=COLORS["charcoal"]))
    return fig

def plot_heading_error_chart_plotly(df, step=50):
    df_h = df.dropna(subset=['Heading error']).reset_index(drop=True)
    if df_h.empty:
        fig = go.Figure()
        fig.add_annotation(text="No heading error data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=14))
        fig.update_layout(title="Heading Error Over Record Index", height=360,
                           plot_bgcolor=COLORS["card"], paper_bgcolor=COLORS["card"], font=PLOTLY_FONT)
        return fig
    mean_he = float(np.mean(df_h['Heading error']))  
    df_sampled = df_h.iloc[::step].reset_index(drop=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_sampled.index.tolist(), y=df_sampled['Heading error'].tolist(), mode="lines",
        line=dict(color=COLORS["charcoal"], width=1.2),
        fill="tozeroy", fillcolor=_hex_to_rgba(COLORS["steel"], 0.10),
        name="Heading Error (°)", hovertemplate="Sample %{x}: %{y:.2f}°<extra></extra>"))
    fig.add_hline(y=0, line_color=COLORS["steel"], line_width=0.8)
    fig.add_hline(y=mean_he, line_dash="dash", line_color=COLORS["orange"], line_width=2,
                  annotation_text=f"Mean = {mean_he:.2f}°", annotation_font_color=COLORS["charcoal"])
    fig.update_layout(
        title=f" Heading Error ",
        height=360, margin=dict(l=40, r=20, t=55, b=40),
        plot_bgcolor=COLORS["card"], paper_bgcolor=COLORS["card"], font=PLOTLY_FONT,
        showlegend=False)
    fig.update_xaxes(title=f" Data ", gridcolor=COLORS["border"], zeroline=False, tickfont=dict(color=COLORS["charcoal"]))
    fig.update_yaxes(title="Heading Error (°)", gridcolor=COLORS["border"], zeroline=False, tickfont=dict(color=COLORS["charcoal"]))
    return fig

# PDF REPORT HELPER CHARTS
def plot_velocity_chart(df):
    """Plot raw velocity from the log data."""
    fig, ax = plt.subplots(figsize=(11, 4))
    if 'velocity' not in df.columns:
        ax.text(0.5, 0.5, "No 'velocity' field found in log data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Raw Velocity")
        return fig
    df_v = df.dropna(subset=['velocity']).reset_index(drop=True)
    if df_v.empty:
        ax.text(0.5, 0.5, "No velocity data available", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Raw Velocity")
        return fig
    vel = df_v['velocity'].values
    ax.plot(range(len(vel)), vel, color=COLORS["orange"], linewidth=1.2, alpha=0.85)
    ax.axhline(np.mean(vel), color=COLORS["amber_dark"], linestyle='--', linewidth=1.5,
               label=f"Mean = {np.mean(vel):.4f}")
    ax.set_title("Raw Velocity", fontweight='bold')
    ax.set_xlabel("Record Index")
    ax.set_ylabel("Velocity")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig

def plot_heading_error_chart(df):
    """Plot heading error over record index."""
    fig, ax = plt.subplots(figsize=(11, 4))
    df_h = df.dropna(subset=['Heading error']).reset_index(drop=True)
    if df_h.empty:
        ax.text(0.5, 0.5, "No heading error data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Heading Error Over Time")
        return fig
    ax.plot(df_h.index, df_h['Heading error'].values, color=COLORS["charcoal"], linewidth=1.0, alpha=0.75)
    ax.axhline(0, color=COLORS["steel"], linestyle='-', linewidth=0.8)
    mean_he = float(np.mean(df_h['Heading error']))
    ax.axhline(mean_he, color=COLORS["orange"], linestyle='--', linewidth=1.5,
               label=f"Mean = {mean_he:.2f}°")
    ax.set_title("Heading Error Over Record Index", fontweight='bold')
    ax.set_xlabel("Record Index")
    ax.set_ylabel("Heading Error (°)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig

# PDF REPORT GENERATOR
def generate_pdf_report(metrics_fig, fig1, fig2, fig3, fig4, fig5, fig6, fig_overall, fig_filtered, fig_sliced, report_df, fig_velocity=None, fig_heading_error=None):
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")

    with PdfPages(temp_pdf.name) as pdf:
        cover_fig = plt.figure(figsize=(11, 8))
        plt.axis('off')
        plt.text(0.5, 0.8, "Vehicle Path Analytics Report", ha='center', fontsize=24, fontweight='bold')
        plt.text(0.5, 0.7, "Streamlit Dashboard", ha='center', fontsize=14)
        pdf.savefig(cover_fig)
        plt.close(cover_fig)

        figs = [metrics_fig, fig_velocity, fig_heading_error, fig1, fig2, fig3, fig4, fig5, fig6, 
                 fig_overall, fig_filtered, fig_sliced]
        
        for fig in figs:
            if fig is not None:
                pdf.savefig(fig)
                plt.close(fig)

        if report_df is not None:
            table_fig, ax = plt.subplots(figsize=(11, 8))
            ax.axis('off')
            ax.set_title("Summary Metrics Data", fontsize=16, fontweight='bold', pad=20)
            table_data = []
            for col in report_df.columns:
                val = report_df[col].iloc[0]
                formatted_val = f"{val:.4f}" if isinstance(val, (float, int)) and not isinstance(val, bool) else str(val)
                table_data.append([col, formatted_val])
            table = ax.table(
                cellText=table_data,
                colLabels=["Metric", "Value"],
                loc='center',
                cellLoc='left',
                colWidths=[0.5, 0.3])
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.8) 

            pdf.savefig(table_fig)
            plt.close(table_fig)

    return temp_pdf.name

# MAIN DASHBOARD
if uploaded_file:
    df_clean, total_raw_records = load_and_preprocess_log_data(uploaded_file)

    initial_count = len(df_clean) 
    df_clean = remove_coordinate_outliers(df_clean, columns=['Latitude', 'Longitude'], method='iqr', factor=1.5)
    removed_count = initial_count - len(df_clean)
    
    if removed_count > 0:
        st.sidebar.warning(f"🚨 Filtered out {removed_count} trajectory outliers.")
    st.success(f"Loaded {len(df_clean)} valid records")

    if df_clean.empty:
        st.error("No valid GPS records found in this file after cleaning. Please check the log format.")
        st.stop()

    (filtered_checkpoints,all_ctes,all_headings,checkpoint_summary_df) = analyze_checkpoints(df_clean)

    mean_cte, mean_heading, conf_cte, conf_heading = calculate_metrics(all_ctes, all_headings, confidence_threshold)
    filtered_df = get_filtered_path_data(df_clean, filtered_checkpoints)
    sliced_df = get_sliced_path_data(filtered_df, filtered_checkpoints)
    sliced_plot_df = sliced_df[
    sliced_df['Heading error'].abs() < 7]
 
    # CTE ANALYSIS
    overall_conf, overall_req = run_cte_analysis(df_clean["CTE"].dropna().values)
    filtered_conf, filtered_req = run_cte_analysis(filtered_df["CTE"].dropna().values if not filtered_df.empty else [])
    sliced_conf, sliced_req = run_cte_analysis(sliced_plot_df["CTE"].dropna().values if not sliced_df.empty else [])
    
  # Analysis
    (filtered_checkpoints,all_ctes,all_headings,checkpoint_summary_df ) = analyze_checkpoints(df_clean)
    (mean_abs_cte,mean_abs_heading,confidence_cte,confidence_heading) = calculate_metrics(all_ctes,all_headings,confidence_threshold)
    filtered_df = get_filtered_path_data(df_clean,filtered_checkpoints)
    sliced_df = get_sliced_path_data(filtered_df,filtered_checkpoints)
  
    # OVERALL METRICS
    overall_cte_mean = np.abs(np.mean(df_clean['CTE'].dropna()))
    overall_heading_mean = np.abs(np.mean(df_clean['Heading error'].dropna()))
    filtered_cte_mean = np.abs(np.mean(filtered_df['CTE'].dropna())) if not filtered_df.empty else 0
    filtered_heading_mean = np.abs(np.mean(filtered_df['Heading error'].dropna())) if not filtered_df.empty else 0
    sliced_cte_mean = np.abs(np.mean( sliced_df['CTE'].dropna())) if not sliced_df.empty else 0
    sliced_heading_mean = np.abs(np.mean( sliced_df['Heading error'].dropna())) if not sliced_df.empty else 0
    overall_sin_theta = get_overall_sin_theta_metric(overall_heading_mean)
    sliced_sin_theta = get_sliced_sin_theta_metric(sliced_heading_mean)

    fig1 = fig2 = fig3 = fig4 = fig5 = fig6 = None
    fig_overall = fig_filtered = fig_sliced = None
  
    # KPI METRICS   
    st.subheader("📌 Key Metrics")
    c1, c2, c3, c4 = st.columns(4)

    avg_cte = float(np.abs(np.mean(sliced_df['CTE'].dropna()))) if not sliced_df.empty else sliced_cte_mean
    avg_heading = np.mean(np.abs( sliced_df['Heading error'].dropna())) if not sliced_df.empty else 0
    total_checkpoints = len(filtered_checkpoints)

    with c1:
        st.metric(
            "📏 Avg CTE ",
            f"{avg_cte:.3f} m")
    with c2:
        st.metric(
            "📏 Avg Heading Error ",
            f"{avg_heading:.3f} deg")
    with c3:
        st.metric(
            "🔖 Total Checkpoints",
            f"{total_checkpoints}")
    with c4:
        st.metric(
            "🗂️ Total Records",
            f"{total_raw_records:,}")

    st.divider()

    # BAR CHARTS   
    st.subheader("📊 Mean Absolute Error Comparison")

    metrics_fig = plot_mean_metrics_bar(
        overall_cte_mean,
        filtered_cte_mean,
        sliced_cte_mean,
        overall_heading_mean,
        filtered_heading_mean,
        sliced_heading_mean)
    for ax in metrics_fig.get_axes():
        ax.grid(False)
    st.plotly_chart(
        plot_mean_metrics_bar_plotly(
            overall_cte_mean, filtered_cte_mean, sliced_cte_mean,
            overall_heading_mean, filtered_heading_mean, sliced_heading_mean),
        use_container_width=True, theme=None)
    st.divider()
    
    # DISTRIBUTIONS    
    st.subheader("📈 Error Distributions")
    overall_cte = df_clean["CTE"].dropna()
    overall_heading = df_clean["Heading error"].dropna()
    
    tabs = st.tabs([
        "Distribution of Overall Vehicle data",
        "Distribution of Filtered Vehicle data",
        "Distribution of Sliced Vehicle data"])
        
    with tabs[0]:
        st.markdown("## 🌍 Overall Distribution")
        col1, col2 = st.columns(2)
        with col1:
          fig1 = plot_distribution(
            overall_cte,
            "CTE Distribution",
            STAGE_COLORS["Overall"]["cte"])
          st.plotly_chart(plot_distribution_plotly(overall_cte, "CTE Distribution", STAGE_COLORS["Overall"]["cte"]), use_container_width=True, theme=None)

        with col2:
          fig2 = plot_distribution(
            overall_heading,
            "Heading Error Distribution",
            STAGE_COLORS["Overall"]["heading"])
          st.plotly_chart(plot_distribution_plotly(overall_heading, "Heading Error Distribution", STAGE_COLORS["Overall"]["heading"]), use_container_width=True, theme=None)
        
    with tabs[1]:
       st.markdown("## 🟡 Filtered Distribution")
       col3, col4 = st.columns(2)
       with col3:
        if not filtered_df.empty:
            fig3 = plot_distribution(
                filtered_df["CTE"].dropna(),
                "CTE Distribution (Filtered)",
                STAGE_COLORS["Filtered"]["cte"])
            st.plotly_chart(plot_distribution_plotly(filtered_df["CTE"].dropna(), "CTE Distribution (Filtered)", STAGE_COLORS["Filtered"]["cte"]), use_container_width=True, theme=None)
        else:
            st.warning("No filtered CTE data")

    with col4:
        if not filtered_df.empty:
            fig4 = plot_distribution(
                filtered_df["Heading error"].dropna(),
                "Heading Error Distribution (Filtered)",
                STAGE_COLORS["Filtered"]["heading"])
            st.plotly_chart(plot_distribution_plotly(filtered_df["Heading error"].dropna(), "Heading Error Distribution (Filtered)", STAGE_COLORS["Filtered"]["heading"]), use_container_width=True, theme=None)
        else:
            st.warning("No filtered heading data")
        
    with tabs[2]:
       st.markdown("## 🟠 Sliced Distribution")
       col5, col6 = st.columns(2)
       with col5:
          if not sliced_df.empty:
            fig5 = plot_distribution(
                sliced_df["CTE"].dropna(),
                "Sliced CTE Distribution",
                STAGE_COLORS["Sliced"]["cte"])
            st.plotly_chart(plot_distribution_plotly(sliced_df["CTE"].dropna(), "Sliced CTE Distribution", STAGE_COLORS["Sliced"]["cte"]), use_container_width=True, theme=None)
          else:
            st.warning("No sliced CTE data available")

       with col6:
          if not sliced_df.empty:
            fig6 = plot_distribution(
                sliced_df["Heading error"].dropna(),
                "Sliced Heading Error Distribution",
                STAGE_COLORS["Sliced"]["heading"])
            st.plotly_chart(plot_distribution_plotly(sliced_df["Heading error"].dropna(), "Sliced Heading Error Distribution", STAGE_COLORS["Sliced"]["heading"]), use_container_width=True, theme=None)
          else:
            st.warning("No sliced heading data available")
    st.divider()
   
    # VEHICLE PATHS  
    st.subheader("🗺️ Vehicle Path Visualization")
    tabs = st.tabs([
        "Overall Vehicle Path",
        "Filtered Vehicle Path",
        "Sliced Vehicle Path"])
    # OVERALL
    with tabs[0]:
        st.markdown("## 🌍 Overall Vehicle Path")
        col1, col2 = st.columns(2)
        with col1:
            fig_overall = plot_vehicle_path_matplotlib(
                df_clean,
                "Overall Vehicle Path",
                STAGE_COLORS["Overall"]["cte"],
                segmented=False)
            st.plotly_chart(
                plot_vehicle_path_plotly(df_clean, "Overall Vehicle Path", STAGE_COLORS["Overall"]["cte"], segmented=False),
                use_container_width=True, theme=None)

        with col2:

            overall_map = create_folium_map(
                df_clean,
                "Overall Vehicle Path",
                STAGE_COLORS["Overall"]["cte"],
                segmented=False
            )

            st_folium(
                overall_map,
                width=700,
                height=500
            )

    # FILTERED
 
    with tabs[1]:
        st.markdown("## 🟡 Filtered Vehicle Path")
        if not filtered_df.empty:
            col1, col2 = st.columns(2)
            with col1:
                fig_filtered = plot_vehicle_path_matplotlib(filtered_df,"Filtered Vehicle Path",STAGE_COLORS["Filtered"]["cte"],segmented=True)
                st.plotly_chart(
                    plot_vehicle_path_plotly(filtered_df, "Filtered Vehicle Path", STAGE_COLORS["Filtered"]["cte"], segmented=True),
                    use_container_width=True, theme=None)

            with col2:
                filtered_map = create_folium_map(filtered_df,"Filtered Vehicle Path", STAGE_COLORS["Filtered"]["cte"],segmented=True)
                st_folium(
                    filtered_map,width=700,height=500)  

        else:
            st.warning("No filtered data available.")
    
    # SLICED
    with tabs[2]:
        st.markdown("## 🟠 Sliced Vehicle Path")

        if not sliced_df.empty:
            col1, col2 = st.columns(2)

            with col1:
                fig_sliced= plot_vehicle_path_matplotlib(sliced_plot_df, "Sliced vehicle path",STAGE_COLORS["Sliced"]["cte"],segmented=True)
                st.plotly_chart(
                    plot_vehicle_path_plotly(sliced_plot_df, "Sliced Vehicle Path", STAGE_COLORS["Sliced"]["cte"], segmented=True),
                    use_container_width=True, theme=None)

            with col2:
                sliced_map = create_folium_map(sliced_plot_df,"Sliced Vehicle Path",STAGE_COLORS["Sliced"]["cte"],segmented=True)
                st_folium(sliced_map,width=700,height=500)
        else:
            st.warning("No sliced data available.")
    st.divider()

    # VELOCITY & HEADING ERROR CHARTS
    st.subheader("📡 Velocity & Heading Error")
    vcol1, vcol2 = st.columns(2)
    with vcol1:
        st.plotly_chart(plot_velocity_chart_plotly(df_clean, step=20), use_container_width=True, theme=None)
    with vcol2:
        st.plotly_chart(plot_heading_error_chart_plotly(df_clean, step=20), use_container_width=True, theme=None)
    st.divider()

# DOWNLOAD REPORT  
    st.subheader("⬇️ Export Analysis")
    report_df = pd.DataFrame([{
        "Overall Mean Abs CTE": overall_cte_mean,
        "Filtered Mean Abs CTE": filtered_cte_mean,
        "Sliced Mean Abs CTE": sliced_cte_mean,
        "Overall Sin Theta Metric": overall_sin_theta,
        "Sliced Sin Theta Metric": sliced_sin_theta,
        "Overall Mean Heading Error": overall_heading_mean,
        "Filtered Mean Heading Error": filtered_heading_mean,
        "Sliced Mean Heading Error": sliced_heading_mean,
        "CTE Confidence": confidence_cte,
        "Heading Confidence": confidence_heading}])
    fig_velocity = plot_velocity_chart(df_clean)
    fig_heading_error_chart = plot_heading_error_chart(df_clean)

 # GENERATE PDF REPORT
    pdf_path = generate_pdf_report(
        metrics_fig,
        fig1,
        fig2,
        fig3,
        fig4,
        fig5,
        fig6,
        fig_overall,
        fig_filtered,
        fig_sliced,
        report_df,
        fig_velocity=fig_velocity,
        fig_heading_error=fig_heading_error_chart)
    csv = report_df.to_csv(index=False).encode("utf-8")
    with open(pdf_path, "rb") as pdf_file:
        st.download_button(
            label="📄 Download Full PDF Report",
            data=pdf_file,
            file_name="vehicle_analytics_report.pdf",
            mime="application/pdf")
else:
    st.info("👈 Upload a vehicle log file from the sidebar to begin analysis.")