import torch

def ensemble_model(mini_batch, models):
    outputs = []
    for model in models:
        out = model(mini_batch)
        outputs.append(out)
    return torch.mean(torch.stack(outputs), dim=0)
