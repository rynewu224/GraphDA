import torch
import torch.nn as nn
import torch.nn.functional as F


def DGDA_loss(res, labels, adj, domain, weights, manipulate=False, dadj=None):

    class_weight, recons_weight, beta, ent_weight, d_w, y_w, m_w = weights

    # Reconstruction loss
    recon_loss = recons_weight * recons_loss(res['a_recons'], adj)
    if manipulate:
        recon_loss += m_w * recons_loss(res['m_recons'], dadj)

    kld = kl_loss(res['dmu'], res['dlv'])
    kly = kl_loss(res['ymu'], res['ylv'])
    klm = kl_loss(res['mmu'], res['mlv'])
    kld = kld + kly + klm

    ent_loss = max_entropy(res['d']) + max_entropy(res['y']) + max_entropy(res['m'])

    if domain == 0:
        class_loss = F.cross_entropy(input=res['cls_output'], target=labels, weight=class_weight)
        domain_labels = torch.zeros_like(labels).float()
    else:
        class_loss = torch.zeros(())
        domain_labels = torch.ones_like(labels).float()

    domain_loss = F.binary_cross_entropy_with_logits(input=res['dom_output'].view(-1), target=domain_labels)

    loss = recon_loss + beta * kld + y_w * class_loss + d_w * domain_loss + ent_weight * ent_loss

    loss = torch.maximum(loss, torch.zeros_like(loss))

    return loss


def DGDA_m_loss(res, labels, adj, domain, weights):
    class_weight, recons_weight, beta, ent_weight, d_w, y_w, m_w = weights
    recon_loss = recons_weight * recons_loss(res['a_recons'], adj)
    kld = kl_loss(res['dmu'], res['dlv'])
    kly = kl_loss(res['ymu'], res['ylv'])
    kld = kld + kly
    ent_loss = max_entropy(res['d']) + max_entropy(res['y'])

    if domain == 0:
        class_loss = F.cross_entropy(input=res['cls_output'], target=labels, weight=class_weight)
        domain_labels = torch.zeros_like(labels).float()
    else:
        class_loss = torch.zeros(())
        domain_labels = torch.ones_like(labels).float()

    domain_loss = F.binary_cross_entropy_with_logits(input=res['dom_output'].view(-1), target=domain_labels)
    loss = recon_loss + beta * kld + y_w * class_loss + d_w * domain_loss + ent_weight * ent_loss
    loss = torch.maximum(loss, torch.zeros_like(loss))

    return loss


def DSR_loss(res, labels, adj, domain, weights):
    class_weight, recons_weight, beta, d_w, y_w = weights

    # Reconstruction loss
    recon_loss = recons_weight * recons_loss(res['a_recons'], adj)
    kld = kl_loss(res['dmu'], res['dlv'])
    kly = kl_loss(res['ymu'], res['ylv'])
    kld = kld + kly

    if domain == 0:
        domain_labels = torch.zeros_like(labels).float()
        sem_cls_loss = F.cross_entropy(input=res['sem_cls'], target=labels, weight=class_weight)
        sem_dom_loss = F.binary_cross_entropy_with_logits(input=res['sem_dom'].view(-1), target=domain_labels)
        dom_cls_loss = max_entropy(res['dom_cls'])
        dom_dom_loss = F.binary_cross_entropy_with_logits(input=res['dom_dom'].view(-1), target=domain_labels)

    else:
        domain_labels = torch.ones_like(labels).float()
        sem_cls_loss = 0
        sem_dom_loss = F.binary_cross_entropy_with_logits(input=res['sem_dom'].view(-1), target=domain_labels)
        dom_cls_loss = max_entropy(res['dom_cls'])
        dom_dom_loss = F.binary_cross_entropy_with_logits(input=res['dom_dom'].view(-1), target=domain_labels)

    loss = recon_loss + beta * kld + sem_cls_loss + sem_dom_loss + dom_cls_loss + dom_dom_loss

    return loss


def DIVA_loss(res, labels, adj, domain, weights):
    class_weight, recons_weight, beta, d_w, y_w = weights

    # Reconstruction loss
    rl = recons_weight * recons_loss(res['a_recons'], adj)

    kld = kl_loss(res['dmu'], res['dlv'])
    kly = kl_loss(res['ymu'], res['ylv'])
    klm = kl_loss(res['mmu'], res['mlv'])

    kld = kld + kly + klm

    if domain == 0:
        cl = F.cross_entropy(input=res['cls_output'], target=labels, weight=class_weight)
        domain_labels = torch.zeros_like(labels).float()
    else:
        cl = torch.zeros(())
        domain_labels = torch.ones_like(labels).float()

    dl = F.binary_cross_entropy_with_logits(input=res['dom_output'].view(-1), target=domain_labels)

    loss = rl + beta * kld + y_w * cl + d_w * dl

    return loss


def DANN_loss(res, labels, domain, class_weight, dw, yw):
    if domain == 0:
        cl = F.cross_entropy(input=res['cls_output'], target=labels, weight=class_weight)
        domain_labels = torch.zeros_like(labels).float()
    else:
        cl = 0.0
        domain_labels = torch.ones_like(labels).float()

    dl = F.binary_cross_entropy_with_logits(input=res['dom_output'].view(-1), target=domain_labels)

    loss = yw * cl + dw * dl

    return loss


def MDD_loss(src_res, tar_res, labels_source, class_weight, src_weight=1.0):
    class_criterion = nn.CrossEntropyLoss(weight=class_weight)

    # _, outputs, _, outputs_adv = self.c_net(inputs)
    n_src = labels_source.size(0)
    n_tar = tar_res['cls_output'].size(0)
    outputs = torch.cat([src_res['cls_output'], tar_res['cls_output']], dim=0)
    # outputs_adv = torch.cat([src_res['adv_output'], tar_res['adv_output']], dim=0)

    classifier_loss = class_criterion(input=src_res['cls_output'], target=labels_source)

    target_adv = outputs.max(1)[1]
    target_adv_src = target_adv.narrow(0, 0, n_src)
    target_adv_tgt = target_adv.narrow(0, n_src, n_tar)

    classifier_loss_adv_src = class_criterion(input=src_res['adv_output'], target=target_adv_src)

    logloss_tgt = torch.log(1.0 - F.softmax(tar_res['adv_output'], dim=1))
    classifier_loss_adv_tgt = F.nll_loss(logloss_tgt, target_adv_tgt)

    transfer_loss = src_weight * classifier_loss_adv_src + classifier_loss_adv_tgt

    total_loss = classifier_loss + transfer_loss

    return total_loss


def recons_loss(recons, adjs):
    batch_size, n_node, _ = recons.shape
    total_node = batch_size * n_node * n_node
    n_edges = adjs.sum()
    device = adjs.device

    if n_edges == 0:  # no positive edges
        pos_weight = torch.zeros(()).to(device)
    else:
        pos_weight = float(total_node - n_edges) / n_edges

    norm = float(total_node) / (2 * (total_node - n_edges))

    rl = norm * F.binary_cross_entropy_with_logits(input=recons, target=adjs, pos_weight=pos_weight, reduction='mean')

    rl = torch.maximum(rl, torch.zeros_like(rl))

    return rl


def kl_loss(mu, lv):
    n_node = mu.shape[1]
    kld = -0.5 / n_node * torch.mean(torch.sum(1 + 2 * lv - mu.pow(2) - lv.exp().pow(2), dim=-1))
    return kld


def max_entropy(x):
    ent = 0.693148 + torch.mean(torch.sigmoid(x) * F.logsigmoid(x))
    return ent


def l1_loss(x):
    return x.abs().mean()


def learning_rate_adjust(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def learning_rate_decay(optimizer, decay_rate=0.99):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate


def clip_gradient(optimizer, grad_clip=0.1):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

