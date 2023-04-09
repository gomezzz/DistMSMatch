import torch

def aggregate_models(local_sd, weights, paths):
    """Aggregate models with a reference model
    
    Args:
        local_sd (dict): reference model
        weights (torch): weights for the different models (reference model is the first)
        paths (list): list of paths to the models to be aggregated
    """
    # Load the models
    local_models = []
    for path in paths:
        try:
            local_models.append(
                torch.load(path).to("cpu").state_dict()
            )
        except:
            print(f"{path} was not successfully loaded", flush=True)
    
    # Aggregate the models
    for key in local_sd:
        local_sd[key] = weights[0] * local_sd[key] + sum([sd[key] * weights[i+1] for i, sd in enumerate(local_models)])
    
    return local_sd