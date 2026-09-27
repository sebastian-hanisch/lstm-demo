import lstm_constants as C
import lstm_evaluation as ev


def test_all_presets_have_valid_settings():
    for key, preset in C.PRESETS.items():
        assert C.T_MIN <= preset["T"] <= C.T_MAX
        assert C.HIDDEN_MIN <= preset["hidden"] <= C.HIDDEN_MAX


def test_preset_kurz_both_models_succeed():
    p = C.PRESETS["kurz"]
    settings = ev.Settings(p["T"], p["hidden"], p["n_train"], p["n_test"], p["noise"],
                           p["eta"], p["epochs"], p["seed"])
    out = ev.analyse(settings)
    assert out["rnn"]["test_accuracy"] >= 0.9
    assert out["lstm"]["test_accuracy"] >= 0.9


def test_preset_lang_kaltstart_runs_and_produces_valid_accuracies():
    """Nur ein Rauchtest: der Einzel-Seed bei T=150 Kaltstart ist chaotisch
    seed-/plattformabhaengig (siehe test_model.py). Die robuste Aussage ueber
    mehrere Seeds steht in test_claims.py."""
    p = C.PRESETS["lang_kaltstart"]
    settings = ev.Settings(p["T"], p["hidden"], p["n_train"], p["n_test"], p["noise"],
                           p["eta"], p["epochs"], p["seed"])
    out = ev.analyse(settings)
    assert 0.0 <= out["rnn"]["test_accuracy"] <= 1.0
    assert 0.0 <= out["lstm"]["test_accuracy"] <= 1.0
