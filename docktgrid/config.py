import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def is_using_gpu():
    return torch.cuda.is_available()
