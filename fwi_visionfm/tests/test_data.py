from fwi_visionfm.pasd.data import synthetic_openfwi_like


def test_synthetic_dataset_shapes():
    records, velocities = synthetic_openfwi_like(n=4, shots=3, time=32, receivers=12, model_size=16)
    assert records.shape == (4, 3, 32, 12)
    assert velocities.shape == (4, 16, 16)
