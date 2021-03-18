import numpy as np
import torch
import torch.random
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, roc_auc_score
from typing import Optional
from torch.optim.optimizer import Optimizer


def modelsize(model, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

def get_metrics(y_true, y_score):
    prc, rec, thr = precision_recall_curve(y_true, y_score)
    f1s = 2 * prc * rec / (prc + rec)
    f1s = f1s[:-1]
    thr = thr[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thr[np.argmax(f1s)]
    y_score = np.array(y_score)
    y_pred = np.zeros_like(y_score)
    y_pred[y_score > best_thr] = 1
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_score)
    return auc, f1

def vectorized_sym_norm(adjs):
    adjs += torch.eye(adjs.shape[1], device=adjs.device).unsqueeze(0).expand_as(adjs)
    inv_sqrt_D = 1.0 / adjs.sum(dim=-1, keepdim=True).sqrt()  # B x N x 1
    inv_sqrt_D[torch.isinf(inv_sqrt_D)] = 0.0
    normalized_adjs = (inv_sqrt_D * adjs) * inv_sqrt_D.transpose(1, 2)
    return normalized_adjs

def drop_edges(adj, drop_rate=0.0, add_rate=0.0):
    bs, N, _ = adj.shape
    n_edges = adj.sum()
    sparsity = (n_edges + bs * N) / (bs * N * N)
    nadj = adj.clone()

    if drop_rate > 0.0:
        drop_mask = torch.bernoulli(adj, p=drop_rate)
        nadj -= drop_mask
        nadj[nadj < 0] = 0

    if add_rate > 0.0:
        add_mask = torch.bernoulli(adj, p=sparsity * add_rate)
        nadj += add_mask

    nadj = nadj.tril() + nadj.tril().permute(0, 2, 1)

    I = torch.eye(N).expand((bs, N, N)).to(adj.device)
    nadj += I
    nadj[nadj < 0] = 0
    nadj[nadj > 0] = 1

    dadj = adj - nadj
    dadj[dadj != 0] = 1
    dadj -= I
    dadj[dadj < 0] = 0

    return nadj, dadj

def drop_nodes(adj, drop_rate=0.0):
    bs, N, _ = adj.shape
    nadj = adj.clone()

    if drop_rate > 0.0:
        drop_size = int(np.ceil(N * drop_rate))
        drop_index = torch.randint(low=0, high=N-1, size=(bs, drop_size))
        for i, drop_index in enumerate(drop_index):
            nadj[i, :, drop_index] = 0
            nadj[i, drop_index, :] = 0

    I = torch.eye(N).expand((bs, N, N)).to(adj.device)
    nadj += I

    # dadj
    dadj = adj - nadj
    dadj[dadj != 0] = 1
    dadj -= I
    dadj[dadj < 0] = 0

    return nadj, dadj

def get_logger(filename=None):
    """
    logging configuration: set the basic configuration of the logging system
    :param filename:
    :return:
    """

    import logging
    import sys
    log_formatter = logging.Formatter(fmt='%(asctime)s [%(levelname)-5.5s] %(message)s',
                                      datefmt='%Y-%b-%d %H:%M')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(log_formatter)
    std_handler.setLevel(logging.DEBUG)
    logger.addHandler(std_handler)

    return logger

class StepwiseLR:

    def __init__(self, optimizer: Optimizer, init_lr: Optional[float],
                 gamma: Optional[float], decay_rate: Optional[float]):
        """
            A lr_scheduler that update learning rate using the following schedule:

            .. math::
                \text{lr} = \text{init_lr} \times \text{lr_mult} \times (1+\gamma i)^{-p},

            where `i` is the iteration steps.

            Parameters:
                - **optimizer**: Optimizer
                - **init_lr** (float, optional): initial learning rate. Default: 0.01
                - **gamma** (float, optional): :math:`\gamma`. Default: 0.001
                - **decay_rate** (float, optional): :math:`p` . Default: 0.75
        """
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0

    def get_lr(self) -> float:
        lr = self.init_lr * (1 + self.gamma * self.iter_num) ** (-self.decay_rate)
        return lr

    def step(self):
        lr = self.get_lr()
        self.iter_num += 1
        for param_group in self.optimizer.param_groups:
            if "lr_mult" not in param_group:
                param_group["lr_mult"] = 1
            param_group['lr'] = lr * param_group["lr_mult"]
