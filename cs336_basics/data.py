import torch
import numpy.typing as npt
import os
from typing import IO, BinaryIO


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """

    import numpy as np

    x = dataset
    m = context_length
    starts = np.random.randint(0, len(x) - context_length, size=(batch_size,))
    inputs = np.stack([x[i : i + m] for i in starts])  # m = context_length
    targets = np.stack([x[i + 1 : i + m + 1] for i in starts])
    return torch.tensor(inputs, device=device), torch.tensor(targets, device=device)
    # import random
    # data_in = torch.empty(batch_size, context_length, dtype=int, device=device)
    # data_target = torch.empty(batch_size, context_length, dtype=int, device=device)
    # for k in range(batch_size):
    #     i = random.randint(0, len(dataset) - context_length - 1)
    #     data_in[k, :] = torch.from_numpy(dataset[i : i + context_length])
    #     data_target[k, :] = torch.from_numpy(dataset[i + 1 : i + 1 + context_length])
    # return (data_in, data_target)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    obj = {"it": iteration}
    obj["model"] = model.state_dict()
    obj["optimizer"] = optimizer.state_dict()
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    obj = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["it"]
