import torch
import torchvision.transforms.v2.functional as F


def preprocess(X, size, device):
    H, W = X.shape[-2:]

    X = torch.from_numpy(X).to(device)

    if X.ndim == 3:
        X = X.unsqueeze(1)

    match X.dtype:
        case torch.uint8:
            X = X.float() / 255.0
        case torch.bool:
            X = X.float()
        case torch.float:
            pass
        case _:
            raise TypeError(f"unexpected dtype: {X.dtype}")

    X = F.pad(X, [0, 0, max(0, H - W), max(0, W - H)])
    X = F.resize(X, size, antialias=True)

    return X.cpu()


def postprocess(X: torch.Tensor, H, W):
    X = F.resize(X, max(H, W), antialias=True)
    X = F.crop(X, 0, 0, H, W)
    return X.squeeze().cpu().numpy()
