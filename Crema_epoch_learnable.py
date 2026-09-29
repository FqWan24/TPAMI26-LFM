#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from torch.utils.data import DataLoader,Subset
import torch.optim as optim
from torch.nn import functional as F
import os
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
import json
import numpy as np
import argparse
import random
from sklearn.metrics import f1_score, average_precision_score
from data.template import config
from dataset.CREMA import CramedDataset
from model.AudioVideo import AVClassifier
from utils.utils import (
    create_logger,
    Averager,
    deep_update_dict,
)
from utils.tools import weight_init

def compute_mAP(outputs, labels):
    y_true = labels.cpu().detach().numpy()
    y_pred = outputs.cpu().detach().numpy()
    AP = []
    for i in range(y_true.shape[1]):
        AP.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    return np.mean(AP)

def Alignment(p, q):
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    m = 0.5 * p + 0.5 * q
    kl_p_m = F.kl_div(p.log(), m, reduction='batchmean')
    kl_q_m = F.kl_div(q.log(), m, reduction='batchmean')
    js_score = 0.5 * (kl_p_m + kl_q_m)
    return js_score

def ContrastiveAlignment(a_f, v_f, temperature=0.07):
    """
    Positive:
        audio_i <-> visual_i

    Negative:
        audio_i <-> visual_j, i != j
    """

    if a_f.dim() > 2:
        a_f = a_f.flatten(start_dim=2).mean(dim=-1)

    if v_f.dim() > 2:
        v_f = v_f.flatten(start_dim=2).mean(dim=-1)
    if a_f.size(1) != v_f.size(1):
        raise ValueError(
            f"Feature dimensions must match for contrastive alignment: "
            f"a_f={a_f.shape}, v_f={v_f.shape}"
        )
    a_f = F.normalize(a_f, p=2, dim=1)
    v_f = F.normalize(v_f, p=2, dim=1)
    logits = torch.matmul(a_f, v_f.T) / temperature

    batch_size = a_f.size(0)

    targets = torch.arange(
        batch_size,
        device=a_f.device
    )
    loss_a2v = F.cross_entropy(
        logits,
        targets
    )
    loss_v2a = F.cross_entropy(
        logits.T,
        targets
    )
    loss = 0.5 * (
        loss_a2v + loss_v2a
    )

    return loss


def getAlpha_Learnable_Fitted(epoch):
    # Alpha with Learnable learning is fitted with functions
    coef_alpha1 = [2.04623704e-01, 3.35472727e-03, 1.22989557e-04, -2.92947416e-06, 2.23835486e-08, -5.39717505e-11]
    coef_alpha2 = [7.95376296e-01, -3.35472727e-03, -1.22989557e-04, 2.92947416e-06, -2.23835486e-08, 5.39717505e-11]
    alpha1 = sum(c * (epoch ** i) for i, c in enumerate(coef_alpha1))
    alpha2 = sum(c * (epoch ** i) for i, c in enumerate(coef_alpha2))
    return [alpha1, alpha2]


def getAlpha_Learnable(epoch, valbatch, model, alpha, lr_alpha=1e-4):
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    
    spectrogram, image, y = valbatch
    half_size = spectrogram.size(0) // 2 
    spectrogram = spectrogram[:half_size]
    image = image[:half_size]
    y = y[:half_size]

    image = image.float().cuda()
    y = y.cuda()
    spectrogram = spectrogram.unsqueeze(1).float().cuda()
    
    o_b, o_a, o_v, a_f, v_f = model(spectrogram, image)
    
    loss_alignment = Alignment(o_a, o_v)
    loss_cls = criterion(0.5 * o_v + 0.5 * o_a, y).mean()

    L_total = alpha[0] * loss_cls + alpha[1] * loss_alignment
    theta_grads = torch.autograd.grad(L_total, model.parameters(), create_graph=True)

    cls_grads = torch.autograd.grad(loss_cls, model.parameters(), retain_graph=True, create_graph=True)


    hessian_vector_prod = torch.autograd.grad(
        outputs=theta_grads,
        inputs=model.parameters(),
        grad_outputs=cls_grads,
        retain_graph=True
    )

    with torch.no_grad():
        alpha_grad = hessian_vector_prod[0].mean().item()
        print(alpha_grad)
        
        if epoch % 20 == 0 and epoch != 0:
            lr_alpha = lr_alpha / 2 
        alpha[1] -= lr_alpha * alpha_grad
        alpha[1] = max(0.05, min(0.95, alpha[1])) 
        alpha[0] = 1.0 - alpha[1] 


    return alpha, lr_alpha



def train_audio_video(epoch, train_loader, model, optimizer, logger, cls_k):
    model.train()
    tl = Averager()
    tl_align = Averager()
    tl_fusion = Averager()
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()

    for step, (spectrogram, image, y) in enumerate(tqdm(train_loader)):
        image = image.float().cuda()
        y = y.cuda()
        spectrogram = spectrogram.unsqueeze(1).float().cuda()
        optimizer.zero_grad()
        o_b, o_a, o_v, a_f, v_f = model(spectrogram, image)
        loss_alignment = Alignment(o_a, o_v)
        loss_cls = criterion( 0.5 * o_v+  0.5 *o_a, y).mean()
        loss =   2 * (cls_k[0] * loss_cls + cls_k[1] * loss_alignment)
        loss.backward()
        optimizer.step()
        tl.add(loss.item())
        tl_align.add(loss_alignment.item())
        tl_fusion.add(loss_cls.item())


    loss_ave = tl.item()
    loss_align_all = tl_align.item()
    loss_fusion_all = tl_fusion.item()
    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(('Epoch {epoch:d}: Average Training Loss:{loss_ave:.3f} , Average AlignLoss : {loss_align_all:.3f},Average FusionLoss : {loss_fusion_all:.3f}').format(epoch=epoch, loss_ave=loss_ave, loss_align_all = loss_align_all, loss_fusion_all =loss_fusion_all))

    return model


def val(epoch, val_loader, model, logger):

    model.eval()
    pred_list = []
    pred_list_a = []
    pred_list_v = []
    label_list = []
    soft_pred = []
    soft_pred_a = []
    soft_pred_v = []
    one_hot_label = []
    score_a = 0.0
    score_v = 0.0
    with torch.no_grad():
        for step, (spectrogram, image, y) in enumerate(tqdm(val_loader)):
            label_list = label_list + torch.argmax(y, dim=1).tolist()
            one_hot_label = one_hot_label + y.tolist()
            image = image.cuda()
            y = y.cuda()
            spectrogram = spectrogram.unsqueeze(1).float().cuda()

            o_b ,o_a, o_v,_,_ = model(spectrogram, image)

            soft_pred_a = soft_pred_a + (F.softmax(o_a, dim=1)).tolist()
            soft_pred_v = soft_pred_v + (F.softmax(o_v, dim=1)).tolist()
            soft_pred = soft_pred + (F.softmax(0.5 * o_v+ 0.5 * o_a, dim=1)).tolist()
            pred = (F.softmax( 0.5 * o_v+ 0.5 * o_a, dim=1)).argmax(dim=1)
            pred_a = (F.softmax(o_a, dim=1)).argmax(dim=1)
            pred_v = (F.softmax(o_v, dim=1)).argmax(dim=1)

            pred_list = pred_list + pred.tolist()
            pred_list_a = pred_list_a + pred_a.tolist()
            pred_list_v = pred_list_v + pred_v.tolist()

        f1 = f1_score(label_list, pred_list, average='macro')
        f1_a = f1_score(label_list, pred_list_a, average='macro')
        f1_v = f1_score(label_list, pred_list_v, average='macro')
        correct = sum(1 for x, y in zip(label_list, pred_list) if x == y)
        correct_a = sum(1 for x, y in zip(label_list, pred_list_a) if x == y)
        correct_v = sum(1 for x, y in zip(label_list, pred_list_v) if x == y)
        acc = correct / len(label_list)
        acc_a = correct_a / len(label_list)
        acc_v = correct_v / len(label_list)
        mAP = compute_mAP(torch.Tensor(soft_pred), torch.Tensor(one_hot_label))
        mAP_a = compute_mAP(torch.Tensor(soft_pred_a), torch.Tensor(one_hot_label))
        mAP_v = compute_mAP(torch.Tensor(soft_pred_v), torch.Tensor(one_hot_label))

    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(('Epoch {epoch:d}: f1:{f1:.4f},acc:{acc:.4f},mAP:{mAP:.4f},f1_a:{f1_a:.4f},acc_a:{acc_a:.4f},mAP_a:{mAP_a:.4f},f1_v:{f1_v:.4f},acc_v:{acc_v:.4f},mAP_v:{mAP_v:.4f}').format(epoch=epoch, f1=f1, acc=acc, mAP=mAP,
                                                                                                                                                                                            f1_a=f1_a, acc_a=acc_a, mAP_a=mAP_a,
                                                                                                                                                                                            f1_v=f1_v, acc_v=acc_v, mAP_v=mAP_v))
    return acc, score_a, score_v


if __name__ == '__main__':
    # ----- LOAD PARAM -----
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str, default='./data/crema.json')

    args = parser.parse_args()
    cfg = config

    with open(args.config, "r") as f:
        exp_params = json.load(f)

    cfg = deep_update_dict(exp_params, cfg)

    # ----- SET SEED -----
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['gpu_id']
    # ----- SET LOGGER -----
    local_rank = cfg['train']['local_rank']
    logger, log_file, exp_id = create_logger(cfg, local_rank)

    # ----- SET DATALOADER -----                          
    train_dataset = CramedDataset(config, mode='train')
    test_dataset = CramedDataset(config, mode='test')

    num_samples = len(train_dataset)
    val_indices = torch.randperm(num_samples)[:16] 

    val_dataset = Subset(train_dataset, val_indices)

    train_indices = torch.tensor([i for i in range(num_samples) if i not in val_indices.tolist()])
    train_dataset = Subset(train_dataset, train_indices)

    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True,
                            num_workers=cfg['train']['num_workers'], pin_memory=True)

    val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False,
                            num_workers=8, pin_memory=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg['test']['batch_size'], shuffle=False,
                            num_workers=cfg['test']['num_workers'], pin_memory=True)
    
    val_batch = next(iter(val_loader))
    # ----- MODEL -----
    model = AVClassifier(config=cfg)
    model = model.cuda()
    model.apply(weight_init)


    lr_adjust = config['train']['optimizer']['lr']

    optimizer = optim.SGD(model.parameters(), lr=lr_adjust,
                          momentum=config['train']['optimizer']['momentum'],
                          weight_decay=config['train']['optimizer']['wc'])
    
    lr_alpha = 1e-4

    scheduler = optim.lr_scheduler.StepLR(optimizer, config['train']['lr_scheduler']['patience'], 0.1)
    best_acc = 0
    cls_k = [0.05, 0.95]
    for epoch in range(cfg['train']['epoch_dict']):
        logger.info(('Epoch {epoch:d} is pending...').format(epoch=epoch))
        # cls_k = getAlpha_Learnable_Fitted(epoch)
        cls_k, lr_alpha = getAlpha_Learnable(epoch, val_batch, model, cls_k, lr_alpha)
        scheduler.step()
        model = train_audio_video(epoch, train_loader, model, optimizer, logger, cls_k)
        acc, v_a, v_v = val(epoch, test_loader, model, logger)
        # torch.save(model.state_dict(), f'/data/wfq/LFM/modelWeight/dynamic/Crema_{epoch}_best_model.pth')
