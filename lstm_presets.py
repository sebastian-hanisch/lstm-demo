"""Permalink-Sync (Query-Parameter <-> Session-State) und Presets."""
from dataclasses import dataclass
from typing import Any, Callable

import streamlit as st

import lstm_constants as C


@dataclass(frozen=True)
class SettingSpec:
    key: str
    param: str
    default: Any
    cast: Callable[[str], Any]
    bounds: tuple | None = None


SETTING_SPECS = [
    SettingSpec("T", "t", C.T_DEFAULT, int, (C.T_MIN, C.T_MAX)),
    SettingSpec("hidden", "hidden", C.HIDDEN_DEFAULT, int, (C.HIDDEN_MIN, C.HIDDEN_MAX)),
    SettingSpec("n_train", "ntrain", C.N_TRAIN_DEFAULT, int, (C.N_TRAIN_MIN, C.N_TRAIN_MAX)),
    SettingSpec("n_test", "ntest", C.N_TEST_DEFAULT, int, (C.N_TEST_MIN, C.N_TEST_MAX)),
    SettingSpec("noise", "rauschen", C.NOISE_DEFAULT, float, (C.NOISE_MIN, C.NOISE_MAX)),
    SettingSpec("eta", "eta", C.ETA_DEFAULT, float, (C.ETA_MIN, C.ETA_MAX)),
    SettingSpec("epochs", "epochen", C.EPOCHS_DEFAULT, int, (C.EPOCHS_MIN, C.EPOCHS_MAX)),
    SettingSpec("seed", "seed", C.DEFAULT_SEED, int),
]


def init_session_state_defaults() -> None:
    for spec in SETTING_SPECS:
        if spec.key not in st.session_state:
            st.session_state[spec.key] = spec.default


def load_permalink_settings() -> None:
    params = st.query_params
    for spec in SETTING_SPECS:
        if spec.param in params and spec.key not in st.session_state:
            try:
                value = spec.cast(params[spec.param])
            except (TypeError, ValueError):
                continue
            if spec.bounds is not None:
                lo, hi = spec.bounds
                value = min(max(value, lo), hi)
            st.session_state[spec.key] = value


def sync_query_params(values: dict) -> None:
    for spec in SETTING_SPECS:
        if spec.key in values:
            st.query_params[spec.param] = str(values[spec.key])


def store_from_widget(key: str) -> None:
    st.session_state[key] = st.session_state[f"widget_{key}"]


def apply_preset(preset_key: str) -> None:
    preset = C.PRESETS[preset_key]
    for field in ("T", "hidden", "n_train", "n_test", "noise", "eta", "epochs", "seed"):
        if field in preset:
            st.session_state[field] = preset[field]
            st.session_state[f"widget_{field}"] = preset[field]


def randomize_seed() -> None:
    import random
    new_seed = random.randint(0, 999_999)
    st.session_state["seed"] = new_seed
    st.session_state["widget_seed"] = new_seed
