"""Reine Plotly-Figure-Builder, keine Streamlit-Aufrufe."""
import numpy as np
import plotly.graph_objects as go

import lstm_constants as C

COLOR_RNN = "#d62728"
COLOR_LSTM = "#1f77b4"
COLOR_GRU = "#2ca02c"


def build_sequence_figure(seq: np.ndarray, label: float, title: str = ""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=seq, mode="lines+markers", name="Sequenz",
                             line=dict(color="#7f7f7f"), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=[C.SIGNAL_POS], y=[seq[C.SIGNAL_POS]], mode="markers",
                             name="Signal", marker=dict(size=14, symbol="star",
                             color=COLOR_LSTM if label > 0 else COLOR_RNN)))
    fig.update_layout(
        title=title or f"Sequenz (Signal = {'+1' if label > 0 else '-1'} an Position {C.SIGNAL_POS})",
        xaxis=dict(title="Zeitschritt t", fixedrange=True),
        yaxis=dict(title="Wert", fixedrange=True), height=300,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_t_sweep_figure(rows, title="Kaltstart-Erfolgsquote vs. Sequenzlänge (SGD, kein Curriculum)"):
    Ts = [str(r["T"]) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="RNN", x=Ts, y=[r["rate_rnn"] * 100 for r in rows], marker_color=COLOR_RNN))
    fig.add_trace(go.Bar(name="LSTM", x=Ts, y=[r["rate_lstm"] * 100 for r in rows], marker_color=COLOR_LSTM))
    fig.update_layout(
        title=title, barmode="group", xaxis=dict(title="Sequenzlänge T", fixedrange=True),
        yaxis=dict(title="Erfolgsquote (%)", fixedrange=True, range=[0, 105]),
        height=340, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_gradient_norm_figure(rows, title="Gradientennorm am ERSTEN Zeitschritt: RNN vs. LSTM"):
    Ts = [r["T"] for r in rows]
    rnn_first = [r["rnn"]["norm_first"] for r in rows]
    lstm_first = [r["lstm"]["norm_first"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Ts, y=rnn_first, mode="lines+markers", name="RNN ‖δ₁‖",
                             line=dict(color=COLOR_RNN)))
    fig.add_trace(go.Scatter(x=Ts, y=lstm_first, mode="lines+markers", name="LSTM ‖dc₁‖",
                             line=dict(color=COLOR_LSTM)))
    fig.update_layout(
        title=title, xaxis=dict(title="Sequenzlänge T", fixedrange=True, type="log"),
        yaxis=dict(title="Gradientennorm (log-Skala)", fixedrange=True, type="log"),
        height=340, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_curriculum_comparison_figure(cold_rate, curriculum_rate, model_name,
                                       title="Kaltstart vs. Curriculum-Lernen"):
    fig = go.Figure(go.Bar(
        x=["Kaltstart (direkt bei T)", "Curriculum (schrittweise wachsendes T)"],
        y=[cold_rate * 100, curriculum_rate * 100],
        marker_color=[COLOR_RNN, COLOR_LSTM],
    ))
    fig.update_layout(
        title=f"{title} — {model_name}",
        yaxis=dict(title="Erfolgsquote (%)", range=[0, 105], fixedrange=True),
        xaxis=dict(fixedrange=True), height=320, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig
