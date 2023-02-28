import numpy as np
import torch
from collections import OrderedDict


def model_to_bytestream(model):
    # Return model parameters as a list of NumPy ndarrays
    bytestream = b""  # Empty byte represenation
    for _, val in model.state_dict().items():  # go over each layer
        bytestream += (
            val.cpu().numpy().tobytes()
        )  # convert layer to bytes and concatenate
    return bytestream


def bytestream_to_statedict(bytestream_parameters, model_template):
    # Read out the raw parameters from the bytestream,
    # note that the size information is not inlcuded
    raw_parameters = np.frombuffer(bytestream_parameters, dtype=np.float32)
    # make byte array into list of ndarrays
    parameters = []
    # go over each layer
    for i, val in model_template.state_dict().items():
        layer_shape = tuple(val.size())  # get size of current layer
        # TODO: understand why there are empty layers in efficientnet
        if len(layer_shape) > 0:
            elems = np.prod(layer_shape)  # find number of elements in layer
            layer_params = np.reshape(
                raw_parameters[range(elems)], layer_shape
            )  # make an ndarray the shape of the layer
        else:
            elems = 1
            layer_params = np.zeros(1)
        parameters.append(layer_params)  # add the ndarray to a list
        raw_parameters = np.delete(
            raw_parameters, range(elems)
        )  # delete the used parameters

    params_dict = zip(
        model_template.state_dict().keys(), parameters
    )  # relate each ndarray to a layer
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in params_dict}
    )  # make a dictionary of tensors
    return state_dict
