import torch

def aggregate_models(local_sd, weight, paths):
    """Aggregate models with a reference model
    
    Args:
        local_sd (dict): reference model
        weight (float): weight for weighted average
        paths (list): list of paths to the models to me aggregated
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
        local_sd[key] = weight * local_sd[key] + sum(
            [sd[key] * weight for sd in local_models]
            )
    
    return local_sd