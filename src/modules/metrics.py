import torch


def pearson_correlation(x, y, invert=False):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    if invert:
        return 1 - corr
    return corr
