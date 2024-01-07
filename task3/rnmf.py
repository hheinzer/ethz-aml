import torch
from torch.nn.functional import normalize


def rnmf(data, rank, regval, tol=1e-6, max_iter=1000):
    """https://github.com/neel-dey/robust-nmf/tree/master"""
    device = data.device

    basis = torch.rand(data.shape[0], rank).to(device)
    coeff = torch.rand(rank, data.shape[1]).to(device)
    outlier = torch.rand(data.shape).to(device)

    approx = basis @ coeff + outlier + tol
    err = torch.zeros(max_iter + 1).to(device)
    obj = torch.zeros(max_iter + 1).to(device)

    # Gaussian noise assumption, beta=2
    error = lambda a, b: 0.5 * (torch.norm(a - b, p="fro") ** 2)
    objective = lambda e, r, o: e[-1] + r * torch.sum(torch.sqrt(torch.sum(o**2, dim=0)))

    err[0] = error(data, approx)
    obj[0] = objective(err, regval, outlier)

    for iter in range(max_iter):
        outlier *= data / (approx + regval * normalize(outlier, p=2, dim=0, eps=tol))
        approx = basis @ coeff + outlier + tol

        coeff *= (basis.t() @ data) / (basis.t() @ approx)
        approx = basis @ coeff + outlier + tol

        basis *= (data @ coeff.t()) / (approx @ coeff.t())
        approx = basis @ coeff + outlier + tol

        err[iter + 1] = error(data, approx)
        obj[iter + 1] = objective(err, regval, outlier)

        if torch.abs((obj[iter] - obj[iter + 1]) / obj[iter]) <= tol:
            break

        if iter == (max_iter - 1):
            print("rnmf reached max_iter")

    return basis, coeff, outlier, err[:iter], obj[:iter]
