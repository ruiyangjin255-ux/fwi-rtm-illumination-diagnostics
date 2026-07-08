__all__ = ["OpenFWINpyDataset", "make_openfwi_splits", "compute_openfwi_stats"]


def __getattr__(name):
    if name == "OpenFWINpyDataset":
        from fwi_visionfm.data.openfwi_npy_dataset import OpenFWINpyDataset

        return OpenFWINpyDataset
    if name == "make_openfwi_splits":
        from fwi_visionfm.data.make_openfwi_splits import make_openfwi_splits

        return make_openfwi_splits
    if name == "compute_openfwi_stats":
        from fwi_visionfm.data.compute_openfwi_stats import compute_openfwi_stats

        return compute_openfwi_stats
    raise AttributeError(name)
