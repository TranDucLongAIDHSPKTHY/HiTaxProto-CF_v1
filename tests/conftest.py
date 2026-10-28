import torch


def pytest_configure(config):
    # exact hand/machine comparisons and gradcheck need double precision; real runs use float32
    torch.set_default_dtype(torch.float64)
