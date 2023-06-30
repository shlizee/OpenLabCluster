"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Copyright (c) 2022-2023 University of Washington. Developed in UW NeuroAI Lab by Jingyuan Li.
"""
import torch

def load_model(model_path, model, optimizer, device):
    """Loads the pretrained model

    Inputs:
        model_path: the directory of the saved model
        model: a random initialized model has the same architecture as the saved model
        optimizer: the optimizer used for model training
        device: str, either 'cpu' or 'cuda'
    """
    model_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(model_dict['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(model_dict['optimizer_state_dict'])
        model.train()

    else:
        model.eval()

    return model, optimizer


def save_checkpoint(model, epoch, optimizer, loss, PATH):
    """Saves a checkpoint
    Inputs:
        model: the trained model
        epoch: the number of the current epoch
        optimizer: the optimizer for model training
        loss: the loss of the current epoch
        PATH: a directory to save model
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, PATH)


