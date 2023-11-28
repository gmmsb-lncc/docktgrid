import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def is_using_gpu() -> bool:
    """Check if GPU is available."""
    return torch.cuda.is_available()
