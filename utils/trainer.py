import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import sleep

def _loops(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim,
    loss_criteria: torch.nn,
    num_classes: int = 1,
    train: bool = True
) -> float:

    total_loss = 0
    correct = 0
    device = torch.device("cuda") \
            if torch.cuda.is_available() \
            else torch.device("cpu")
    batch = 0
    length = len(data_loader.dataset)

    with tqdm(data_loader, unit="batch") as tepoch:
        for data, target in tepoch:
            tepoch.set_description("Train" if train else "Test")
            batch += 1
            data = data.to(device)
            target = target.to(device)
            
            if train:
                optimizer.zero_grad()

            out = model(data).to(device)
            loss = loss_criteria(out, target).to(device)
            total_loss += loss.item()

            if (num_classes == 1):
                _, predicted = torch.max(out.data, 1)
                correct += torch.sum(target == predicted).item()
            else:
                predicted = out.argmax(dim=1)
                correct += torch.sum(target == predicted)

            if train:
                loss.backward()
                optimizer.step()

            acc = 100. * correct / length
            guess = f"{correct}/{length}"
            string = f"loss: {100. * loss:.6f}%, accuracy: {acc:.6f}% [{guess}]"
            tepoch.set_postfix_str(string)
            sleep(0.01)

    avg_loss = total_loss / (batch + 1)
    return avg_loss


def train(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim,
    loss_criteria: torch.nn,
    num_classes: int = 1,
) -> float:
    r"""
    Args:
        model (nn.Module): Model use for training
        data_loader (DataLoader): Data loader of training data
        optimizer (torch.optim): Optimizer uses for training
        loss_criteria (torch.nn): Loss function use for training
        num_classes (int): Number of classes (output shape) of dataset
    """

    model.train()

    return _loops(
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        loss_criteria=loss_criteria,
        num_classes=num_classes,
        train=True
    )


def test(
    model: nn.Module,
    data_loader: DataLoader,
    loss_criteria: torch.nn,
    num_classes: int = 1,
) -> float:
    r"""
    Args:
        model (nn.Module): Model use for testing
        data_loader (DataLoader): Data loader of testing data
        loss_criteria (torch.nn): Loss function use for testing
        num_classes (int): Number of classes (output shape) of dataset
    """

    model.eval()
    avg_loss = 0.0
    
    with torch.no_grad():
        avg_loss = _loops(
            model=model,
            data_loader=data_loader,
            optimizer=None,
            loss_criteria=loss_criteria,
            num_classes=num_classes,
            train=False
        )

    return avg_loss