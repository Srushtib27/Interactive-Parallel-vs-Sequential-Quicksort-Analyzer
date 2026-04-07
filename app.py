"""
app.py - Interactive AI-Powered Parallel vs Sequential Quicksort Analyzer
Run: streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import json
import os

from sequential import quicksort, generate_dataset, run_sequential
from utils import (
    run_parallel, simulate_parallel, get_system_info,
    get_cpu_usage, check_mpi_available
)
from analytics import (
    compute_metrics, generate_insights, recommend_config,
    save_result, load_results, export_csv, BenchmarkResult
)
from ml_model import train_model, predict, generate_prediction_curve, compute_r2, load_model

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuickSort Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root theme ── */
:root {
    --bg: #0a0a0f;
    --bg2: #111118;
    --bg3: #18181f;
    --card: #1a1a24;
    --card2: #22222e;
    --border: #2e2e3e;
    --accent1: #7c6af7;   /* violet */
    --accent2: #36d6b5;   /* teal */
    --accent3: #f7b731;   /* amber */
    --accent4: #ff5e7d;   /* pink */
    --text: #e8e8f0;
    --muted: #7a7a8e;
    --font-head: 'Space Mono', monospace;
    --font-body: 'DM Sans', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: var(--font-body) !important; }

/* Main area */
.main .block-container { padding: 1.5rem 2rem !important; max-width: 1400px !important; }

/* Title */
.qs-title {
    font-family: var(--font-head);
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent1) 0%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}
.qs-subtitle {
    font-size: 0.9rem;
    color: var(--muted);
    font-weight: 300;
    margin-bottom: 1.5rem;
}

/* Metric card */
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(124,106,247,0.15); }
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.metric-card.violet::before { background: var(--accent1); }
.metric-card.teal::before   { background: var(--accent2); }
.metric-card.amber::before  { background: var(--accent3); }
.metric-card.pink::before   { background: var(--accent4); }
.metric-label {
    font-size: 0.7rem;
    font-family: var(--font-head);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: var(--font-head);
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
}
.metric-sub { font-size: 0.72rem; color: var(--muted); margin-top: 0.3rem; }

/* Section headers */
.section-header {
    font-family: var(--font-head);
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent1);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 0.8rem;
}

/* Insight card */
.insight-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent1);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-size: 0.85rem;
    margin-bottom: 0.5rem;
    color: var(--text);
    line-height: 1.5;
}

/* Recommendation box */
.rec-box {
    background: linear-gradient(135deg, rgba(124,106,247,0.1), rgba(54,214,181,0.1));
    border: 1px solid var(--accent1);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-size: 0.87rem;
    margin-top: 0.5rem;
}

/* Status badge */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    font-size: 0.7rem;
    font-family: var(--font-head);
    letter-spacing: 0.05em;
    font-weight: 700;
    text-transform: uppercase;
}
.badge-green  { background: rgba(54,214,181,0.15); color: var(--accent2); border: 1px solid var(--accent2); }
.badge-red    { background: rgba(255,94,125,0.15);  color: var(--accent4); border: 1px solid var(--accent4); }
.badge-amber  { background: rgba(247,183,49,0.15);  color: var(--accent3); border: 1px solid var(--accent3); }
.badge-violet { background: rgba(124,106,247,0.15); color: var(--accent1); border: 1px solid var(--accent1); }

/* Sidebar label */
.sidebar-section {
    font-family: var(--font-head);
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 1rem 0 0.3rem;
}

/* Plotly overrides */
.js-plotly-plot .plotly { background: transparent !important; }

/* Hide streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Custom scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Plotly dark theme helper ────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#e8e8f0", size=12),
    legend=dict(bgcolor="rgba(26,26,36,0.8)", bordercolor="#2e2e3e", borderwidth=1),
    xaxis=dict(gridcolor="#2e2e3e", zerolinecolor="#2e2e3e", linecolor="#2e2e3e"),
    yaxis=dict(gridcolor="#2e2e3e", zerolinecolor="#2e2e3e", linecolor="#2e2e3e"),
    margin=dict(l=40, r=20, t=40, b=40),
)
C_VIOLET = "#7c6af7"
C_TEAL   = "#36d6b5"
C_AMBER  = "#f7b731"
C_PINK   = "#ff5e7d"


# ── Session state init ────────────────────────────────────────────────────────
for key, default in [
    ("results", []),
    ("last_result", None),
    ("model", None),
    ("auto_running", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Load persisted results on first run
if not st.session_state.results:
    st.session_state.results = load_results()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="qs-title" style="font-size:1.1rem;">⚡ QuickSort</p>', unsafe_allow_html=True)
    st.markdown('<p class="qs-subtitle">Parallel vs Sequential Analyzer</p>', unsafe_allow_html=True)

    st.markdown('<p class="sidebar-section">Dataset Config</p>', unsafe_allow_html=True)
    n = st.slider("Input Size (n)", 500, 50000, 10000, step=500)
    dataset_type = st.selectbox(
        "Dataset Type",
        ["random", "sorted", "reverse", "nearly_sorted"],
        format_func=lambda x: {
            "random": "🎲 Random",
            "sorted": "📈 Sorted",
            "reverse": "📉 Reverse Sorted",
            "nearly_sorted": "〰️ Nearly Sorted",
        }[x]
    )

    st.markdown('<p class="sidebar-section">Parallel Config</p>', unsafe_allow_html=True)
    max_procs = min(os.cpu_count() or 8, 8)
    processes = st.slider("Processes", 1, max_procs, 2)

    mpi_ok = check_mpi_available()
    if mpi_ok:
        st.markdown('<span class="badge badge-green">MPI Available</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-amber">MPI Simulated</span>', unsafe_allow_html=True)
        st.caption("Install OpenMPI for real parallel execution.")

    st.markdown('<p class="sidebar-section">Actions</p>', unsafe_allow_html=True)
    run_btn       = st.button("▶ Run Analysis", use_container_width=True, type="primary")
    auto_btn      = st.button("🔁 Auto Benchmark", use_container_width=True)
    train_btn     = st.button("🧠 Train ML Model", use_container_width=True)
    clear_btn     = st.button("🗑 Clear History", use_container_width=True)

    # System info
    st.markdown('<p class="sidebar-section">System</p>', unsafe_allow_html=True)
    sys_info = get_system_info()
    st.caption(f"CPU cores: {sys_info.get('cpu_count', 'N/A')}")
    st.caption(f"RAM: {sys_info.get('ram_gb', 'N/A')} GB")
    st.caption(f"Platform: {sys_info.get('platform', '?')}")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="qs-title">⚡ Quicksort Performance Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="qs-subtitle">AI-powered insights · Real MPI parallelism · Live predictions</p>', unsafe_allow_html=True)


# ── Helper: render metric card ────────────────────────────────────────────────
def metric_card(label, value, sub="", color="violet"):
    st.markdown(f"""
    <div class="metric-card {color}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Clear history ─────────────────────────────────────────────────────────────
if clear_btn:
    st.session_state.results = []
    st.session_state.last_result = None
    st.session_state.model = None
    if os.path.exists("benchmark_results.json"):
        os.remove("benchmark_results.json")
    if os.path.exists("ml_model_coeffs.json"):
        os.remove("ml_model_coeffs.json")
    st.success("History cleared.")


# ── Single run ────────────────────────────────────────────────────────────────
def do_run(n_val, procs, dtype):
    progress = st.progress(0, text="Generating dataset…")
    data = generate_dataset(n_val, dtype)
    progress.progress(20, text="Running sequential sort…")

    _, seq_time = run_sequential(data.copy())
    progress.progress(50, text="Running parallel sort…")

    if mpi_ok:
        par_time, err = run_parallel(procs, n_val, dtype)
        if par_time is None:
            st.warning(f"MPI failed: {err}. Falling back to simulation.")
            par_time = simulate_parallel(seq_time, procs, n_val)
    else:
        par_time = simulate_parallel(seq_time, procs, n_val)

    progress.progress(80, text="Computing metrics…")
    result = compute_metrics(n_val, procs, seq_time, par_time, dtype)
    save_result(result)
    st.session_state.results.append(result)
    st.session_state.last_result = result
    progress.progress(100, text="Done!")
    time.sleep(0.3)
    progress.empty()
    return result


if run_btn:
    result = do_run(n, processes, dataset_type)
    st.rerun()


# ── Auto benchmark ────────────────────────────────────────────────────────────
if auto_btn:
    sizes = [1000, 5000, 10000, 20000, 35000, 50000]
    procs_list = [1, 2, 4, min(max_procs, 6)]
    total = len(sizes) * len(procs_list)
    bar = st.progress(0, text="Auto-benchmarking…")
    i = 0
    for sz in sizes:
        for pr in procs_list:
            bar.progress(i / total, text=f"n={sz:,} | {pr} processes")
            do_run(sz, pr, "random")
            i += 1
    bar.progress(1.0, text="Benchmark complete!")
    time.sleep(0.5)
    bar.empty()
    # Auto-train model after benchmark
    m = train_model(st.session_state.results)
    if m:
        st.session_state.model = m
    st.rerun()


# ── Train ML model ─────────────────────────────────────────────────────────────
if train_btn:
    if len(st.session_state.results) < 4:
        st.warning("Need at least 4 benchmark results to train the model. Run more experiments first.")
    else:
        with st.spinner("Training regression model…"):
            m = train_model(st.session_state.results)
            if m:
                st.session_state.model = m
                st.success(f"Model trained on {m['n_samples']} samples!")
            else:
                st.error("Training failed. Insufficient data variety.")


# Load persisted model if not in session
if st.session_state.model is None:
    st.session_state.model = load_model()


# ── Last result section ───────────────────────────────────────────────────────
if st.session_state.last_result:
    r = st.session_state.last_result

    st.markdown('<p class="section-header">📊 Latest Benchmark</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Sequential Time", f"{r.seq_time:.4f}s", f"n={r.n:,}", "violet")
    with c2:
        metric_card("Parallel Time", f"{r.par_time:.4f}s", f"{r.processes} processes", "teal")
    with c3:
        color = "teal" if r.speedup >= 1 else "pink"
        metric_card("Speedup", f"{r.speedup:.2f}×", "par/seq ratio", color)
    with c4:
        eff_color = "teal" if r.efficiency >= 70 else ("amber" if r.efficiency >= 40 else "pink")
        metric_card("Efficiency", f"{r.efficiency:.1f}%", "per-process util", eff_color)

    st.markdown("")
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        metric_card("Throughput (Seq)", f"{r.throughput_seq/1000:.1f}K/s", "elements/sec", "violet")
    with c6:
        metric_card("Throughput (Par)", f"{r.throughput_par/1000:.1f}K/s", "elements/sec", "teal")
    with c7:
        winner = "Parallel" if r.is_parallel_better else "Sequential"
        w_color = "teal" if r.is_parallel_better else "violet"
        metric_card("Winner", winner, f"dtype: {r.dataset_type}", w_color)
    with c8:
        cpu = get_cpu_usage()
        metric_card("CPU Usage", f"{cpu:.0f}%" if cpu >= 0 else "N/A", "current system", "amber")

    # Insights
    st.markdown('<p class="section-header">🤖 AI Insights</p>', unsafe_allow_html=True)
    insights = generate_insights(r, st.session_state.results)
    for ins in insights:
        st.markdown(f'<div class="insight-card">{ins}</div>', unsafe_allow_html=True)

    # ML Prediction block
    if st.session_state.model:
        pred = predict(r.n, r.processes, st.session_state.model)
        if pred:
            st.markdown('<p class="section-header">🔮 ML Prediction vs Actual</p>', unsafe_allow_html=True)
            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                metric_card("Pred. Seq Time", f"{pred['pred_seq']:.4f}s", f"Actual: {r.seq_time:.4f}s", "violet")
            with pc2:
                metric_card("Pred. Par Time", f"{pred['pred_par']:.4f}s", f"Actual: {r.par_time:.4f}s", "teal")
            with pc3:
                metric_card("Pred. Speedup", f"{pred['pred_speedup']:.2f}×", f"Actual: {r.speedup:.2f}×", "amber")

else:
    st.info("👈 Configure settings in the sidebar and click **Run Analysis** to begin.")


# ── Charts ────────────────────────────────────────────────────────────────────
if len(st.session_state.results) >= 2:
    results = st.session_state.results
    df = pd.DataFrame([r.to_dict() for r in results])

    st.markdown('<p class="section-header">📈 Performance Charts</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⏱ Time Comparison",
        "🚀 Speedup vs Processes",
        "💹 Efficiency",
        "📐 Scaling",
        "🔮 ML Predictions",
    ])

    with tab1:
        # Bar chart: seq vs par for last 10 runs
        recent = df.tail(12)
        fig = go.Figure()
        fig.add_bar(
            x=[f"n={int(row.n):,} p={int(row.processes)}" for _, row in recent.iterrows()],
            y=recent["seq_time"],
            name="Sequential",
            marker_color=C_VIOLET,
            opacity=0.85,
        )
        fig.add_bar(
            x=[f"n={int(row.n):,} p={int(row.processes)}" for _, row in recent.iterrows()],
            y=recent["par_time"],
            name="Parallel",
            marker_color=C_TEAL,
            opacity=0.85,
        )
        fig.update_layout(**PLOT_LAYOUT, barmode="group", title="Execution Time: Sequential vs Parallel")
        fig.update_xaxes(tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Speedup grouped by input size
        if "n" in df.columns and "processes" in df.columns:
            fig = go.Figure()
            for n_val in sorted(df["n"].unique()):
                sub = df[df["n"] == n_val].sort_values("processes")
                fig.add_scatter(
                    x=sub["processes"], y=sub["speedup"],
                    mode="lines+markers",
                    name=f"n={int(n_val):,}",
                    line=dict(width=2),
                    marker=dict(size=7),
                )
            # Ideal linear speedup reference
            p_range = list(range(1, int(df["processes"].max()) + 1))
            fig.add_scatter(
                x=p_range, y=p_range,
                mode="lines", name="Ideal (linear)",
                line=dict(dash="dash", color=C_AMBER, width=1.5),
            )
            fig.update_layout(**PLOT_LAYOUT, title="Speedup vs Number of Processes",
                              xaxis_title="Processes", yaxis_title="Speedup (×)")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Efficiency heatmap (n vs processes)
        if len(df["n"].unique()) >= 2 and len(df["processes"].unique()) >= 2:
            pivot = df.groupby(["n", "processes"])["efficiency"].mean().reset_index()
            pivot_wide = pivot.pivot(index="n", columns="processes", values="efficiency")
            fig = go.Figure(go.Heatmap(
                z=pivot_wide.values,
                x=[str(c) for c in pivot_wide.columns],
                y=[f"{int(r):,}" for r in pivot_wide.index],
                colorscale=[[0, "#ff5e7d"], [0.5, "#f7b731"], [1, "#36d6b5"]],
                zmin=0, zmax=100,
                colorbar=dict(title="Efficiency %", tickfont=dict(color="#e8e8f0")),
                text=pivot_wide.values.round(1),
                texttemplate="%{text}%",
            ))
            fig.update_layout(**PLOT_LAYOUT, title="Parallel Efficiency Heatmap (n × Processes)",
                              xaxis_title="Processes", yaxis_title="Input Size")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Line chart fallback
            fig = go.Figure()
            for n_val in sorted(df["n"].unique()):
                sub = df[df["n"] == n_val].sort_values("processes")
                fig.add_scatter(
                    x=sub["processes"], y=sub["efficiency"],
                    mode="lines+markers", name=f"n={int(n_val):,}",
                    line=dict(width=2), marker=dict(size=7),
                )
            fig.add_hline(y=80, line_dash="dash", line_color=C_AMBER,
                          annotation_text="80% threshold", annotation_position="right")
            fig.update_layout(**PLOT_LAYOUT, title="Parallel Efficiency vs Processes",
                              xaxis_title="Processes", yaxis_title="Efficiency (%)")
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        # Scaling: time vs input size per process count
        fig = go.Figure()
        colors = [C_VIOLET, C_TEAL, C_AMBER, C_PINK, "#a78bfa", "#34d399", "#fbbf24", "#f87171"]
        for idx, p_val in enumerate(sorted(df["processes"].unique())):
            sub = df[df["processes"] == p_val].sort_values("n")
            c = colors[idx % len(colors)]
            fig.add_scatter(
                x=sub["n"], y=sub["par_time"],
                mode="lines+markers", name=f"Parallel ({int(p_val)}p)",
                line=dict(color=c, width=2), marker=dict(size=7),
            )
        sub_seq = df.sort_values("n").groupby("n")["seq_time"].mean().reset_index()
        fig.add_scatter(
            x=sub_seq["n"], y=sub_seq["seq_time"],
            mode="lines+markers", name="Sequential",
            line=dict(color="#ffffff", width=2, dash="dot"), marker=dict(size=7),
        )
        fig.update_layout(**PLOT_LAYOUT, title="Scaling: Execution Time vs Input Size",
                          xaxis_title="Input Size (n)", yaxis_title="Time (s)")
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        if st.session_state.model:
            model = st.session_state.model
            r2_scores = compute_r2(results, model)
            st.markdown(
                f'<div class="rec-box">Model R²: '
                f'Sequential <b>{r2_scores["r2_seq"]:.3f}</b> · '
                f'Parallel <b>{r2_scores["r2_par"]:.3f}</b> · '
                f'Trained on <b>{model["n_samples"]}</b> samples</div>',
                unsafe_allow_html=True
            )
            st.markdown("")

            # Prediction curve
            proc_for_pred = st.slider("Processes for prediction curve", 1, max_procs, 2, key="pred_p")
            sizes_curve, pred_seq_curve, pred_par_curve = generate_prediction_curve(proc_for_pred, model)
            fig = go.Figure()
            fig.add_scatter(x=sizes_curve, y=pred_seq_curve, mode="lines",
                            name="Pred. Sequential", line=dict(color=C_VIOLET, dash="dot", width=2))
            fig.add_scatter(x=sizes_curve, y=pred_par_curve, mode="lines",
                            name=f"Pred. Parallel ({proc_for_pred}p)", line=dict(color=C_TEAL, dash="dot", width=2))
            # Actual points
            fig.add_scatter(x=df["n"], y=df["seq_time"], mode="markers",
                            name="Actual Sequential", marker=dict(color=C_VIOLET, size=8, symbol="circle"))
            fig.add_scatter(x=df[df["processes"] == proc_for_pred]["n"],
                            y=df[df["processes"] == proc_for_pred]["par_time"],
                            mode="markers", name=f"Actual Parallel ({proc_for_pred}p)",
                            marker=dict(color=C_TEAL, size=8, symbol="diamond"))
            fig.update_layout(**PLOT_LAYOUT, title="ML Prediction vs Actual (Execution Time)",
                              xaxis_title="Input Size (n)", yaxis_title="Time (s)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Train the ML model (sidebar) to see predictions here.")


# ── Recommendation section ───────────────────────────────────────────────────
if len(st.session_state.results) >= 3:
    st.markdown('<p class="section-header">💡 Smart Recommendations</p>', unsafe_allow_html=True)
    rec = recommend_config(st.session_state.results)
    if rec:
        use_par = rec.get("use_parallel", True)
        badge = '<span class="badge badge-teal">Parallel</span>' if use_par else '<span class="badge badge-violet">Sequential</span>'
        st.markdown(f"""
        <div class="rec-box">
            <b>Recommendation:</b> Use {badge} sorting for your workload.<br>
            <br>
            Optimal processes: <b>{rec['recommended_processes']}</b> ·
            Best at n ≈ <b>{rec['recommended_n']:,}</b> ·
            Best speedup: <b>{rec['best_speedup']:.2f}×</b> ·
            Efficiency: <b>{rec['best_efficiency']:.1f}%</b><br>
            <br>
            <i style="color:#7a7a8e;">{rec['note']}</i>
        </div>
        """, unsafe_allow_html=True)


# ── History table + export ────────────────────────────────────────────────────
if st.session_state.results:
    st.markdown('<p class="section-header">📋 Experiment History</p>', unsafe_allow_html=True)
    df_hist = pd.DataFrame([r.to_dict() for r in st.session_state.results])
    df_display = df_hist[[
        "timestamp", "n", "processes", "dataset_type",
        "seq_time", "par_time", "speedup", "efficiency",
        "throughput_seq", "throughput_par", "is_parallel_better"
    ]].copy()
    df_display.columns = [
        "Timestamp", "N", "Procs", "Type",
        "Seq Time (s)", "Par Time (s)", "Speedup ×", "Efficiency %",
        "Tput Seq", "Tput Par", "Par Better?"
    ]
    st.dataframe(df_display, use_container_width=True, height=260)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv_data = df_hist.to_csv(index=False)
        st.download_button(
            "⬇ Download CSV",
            csv_data,
            file_name="benchmark_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_dl2:
        json_data = json.dumps([r.to_dict() for r in st.session_state.results], indent=2)
        st.download_button(
            "⬇ Download JSON",
            json_data,
            file_name="benchmark_results.json",
            mime="application/json",
            use_container_width=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:3rem; color:#3a3a4e; font-size:0.72rem; font-family:'Space Mono',monospace;">
  QuickSort Analyzer · Parallel + Sequential · AI-Powered Insights
</div>
""", unsafe_allow_html=True)
