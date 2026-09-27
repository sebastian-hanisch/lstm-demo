from streamlit.testing.v1 import AppTest

# Deutlich hoeherer Timeout als in den Geschwister-Repos: der Kaltstart-
# T-Sweep (6 Inits x 4 Sequenzlaengen bis T=150, RNN UND LSTM) braucht auf
# einem langsamen CI-Runner mehrere Minuten (siehe
# feedback_expensive_sweep_needs_visible_spinner.md). 300s reichte auf der
# Linux-CI NICHT (322,95s gemessen, CI-Runner ist ca. 1,8x langsamer als
# lokal) - auf 600s mit Sicherheitsmarge erhoeht.
_APP_TIMEOUT = 600


def _fresh():
    at = AppTest.from_file("../app.py", default_timeout=_APP_TIMEOUT)
    at.run()
    return at


def test_app_runs_without_exception():
    at = _fresh()
    assert not at.exception


def test_footer_is_present():
    at = _fresh()
    captions = [c.value for c in at.caption]
    assert any("Sebastian Hanisch" in c and "Kontakt aufnehmen" in c for c in captions)


def test_preset_kurz_shows_high_accuracy_for_both():
    at = _fresh()
    btn = [b for b in at.button if b.label == "Kurze Sequenz — beide gelingen"][0]
    btn.click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert float(metrics["RNN-Testgenauigkeit"].rstrip("%")) >= 90.0
    assert float(metrics["LSTM-Testgenauigkeit"].rstrip("%")) >= 90.0


def test_hidden_slider_extreme_values_do_not_crash():
    at = _fresh()
    h_slider = [s for s in at.slider if s.label.startswith("Verdeckte")][0]
    h_slider.set_value(h_slider.min).run()
    assert not at.exception
    h_slider = [s for s in at.slider if s.label.startswith("Verdeckte")][0]
    h_slider.set_value(h_slider.max).run()
    assert not at.exception
