import numpy as np
import torch


class EarlyStopping:

    def __init__(self, patience=10, verbose=False, delta=0, save_path="checkpoint.pt"):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss=val_loss, model=model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss=val_loss, model=model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save model when validation loss decrease"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).   Saving model ...')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss


class EarlyStoppingF1:

    def __init__(self, patience=10, verbose=False, delta=0, save_path="checkpoint.pt"):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.f1_max = 0.0
        self.delta = delta
        self.save_path = save_path

    def __call__(self, f1, model):
        score = f1

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(f1, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(f1, model)
            self.counter = 0

    def save_checkpoint(self, f1, model):
        """Save model when F1-score increase"""
        if self.verbose:
            print(f'F1-score increased ({self.f1_max:.6f} --> {f1:.6f}).   Saving model ...')
        torch.save(model.state_dict(), self.save_path)
        self.f1_max = f1
