import numpy as np

import lstm_scenario as sc


def test_make_dataset_is_reproducible():
    ds1 = sc.make_dataset(20, T=10, seed=3)
    ds2 = sc.make_dataset(20, T=10, seed=3)
    np.testing.assert_array_equal(ds1.X, ds2.X)
    np.testing.assert_array_equal(ds1.y, ds2.y)


def test_signal_is_at_fixed_position():
    rng = np.random.default_rng(0)
    for _ in range(20):
        seq, label = sc.make_sequence(rng, T=15)
        assert np.sign(seq[0]) in (-1.0, 1.0)


def test_dataset_has_both_labels():
    ds = sc.make_dataset(40, T=10, seed=1)
    assert set(ds.y.tolist()) == {-1.0, 1.0}


def test_dataset_shape_matches_T():
    for T in (2, 10, 50):
        ds = sc.make_dataset(10, T=T, seed=0)
        assert ds.X.shape == (10, T)
