import math
import os
import random
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.core.transforms_interface import DualTransform
from torch import Tensor
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset


# class FocalLoss(nn.Module):
#     """
#     Referenced to
#     - https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html
#     - https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/4
#     - https://github.com/DingKe/pytorch_workplace/blob/master/focalloss/loss.py
#     - https://github.com/Kitsunetic/focal_loss_pytorch
#     """

#     def __init__(self, gamma=2.0, eps=1e-6, reduction="mean"):
#         assert reduction in ["mean", "sum"], f"reduction should be mean or sum not {reduction}."
#         super(FocalLoss, self).__init__()

#         self.gamma = gamma
#         self.eps = eps
#         self.reduction = reduction

#     def forward(self, input, target):
#         plog = F.log_softmax(input, dim=-1)  # TODO: adjusting log directly could occurs gradient exploding???
#         p = torch.exp(plog)
#         focal_weight = (1 - p) ** self.gamma
#         loss = F.nll_loss(focal_weight * plog, target)

#         return loss


class FocalLoss(nn.Module):
    """
    https://dacon.io/competitions/official/235585/codeshare/1796
    """

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # print(self.gamma)
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class ArcFaceLoss(nn.modules.Module):
    """
    https://dacon.io/competitions/official/235585/codeshare/1796
    """

    def __init__(self, gamma=2.0, eps=1e-6, s=45.0, m=0.1, crit="bce", weight=None, reduction="mean"):
        super().__init__()

        self.weight = weight
        self.reduction = reduction

        if crit == "focal":
            self.crit = FocalLoss(gamma=gamma, eps=eps)
        elif crit == "bce":
            self.crit = nn.CrossEntropyLoss(reduction="none")

        if s is None:
            self.s = torch.nn.Parameter(torch.tensor([45.0], requires_grad=True, device="cuda"))
        else:
            self.s = s

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels):

        # logits = logits.float()
        cosine = logits

        # loss divergence problem
        # https://github.com/ronghuaiyang/arcface-pytorch/issues/32#issuecomment-514593801
        # sine = (1.0 - torch.pow(cosine, 2)).sqrt_()
        sine = (1.0 - cosine.pow(2)).clamp_(min=1e-9, max=1).sqrt_()

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        labels2 = torch.zeros_like(cosine)
        labels2.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (labels2 * phi) + ((1.0 - labels2) * cosine)

        s = self.s

        output = output * s
        loss = self.crit(output, labels)

        if self.weight is not None:
            w = self.weight[labels].to(logits.device)

            loss = loss * w
            ### human coding
            class_weights_norm = "batch"
            if class_weights_norm == "batch":
                loss = loss.sum() / w.sum()
            if class_weights_norm == "global":
                loss = loss.mean()
            else:
                loss = loss.mean()

            return loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


class AverageMeter(object):
    """
    AverageMeter, referenced to https://dacon.io/competitions/official/235626/codeshare/1684
    """

    def __init__(self):
        self.sum = 0
        self.cnt = 0
        self.avg = 0

    def update(self, val, n=1):
        if n > 0:
            self.sum += val * n
            self.cnt += n
            self.avg = self.sum / self.cnt

    def get(self):
        return self.avg

    def __call__(self):
        return self.avg


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    p.grad.norm(p=2).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm


class CustomLogger:
    def __init__(self, filename=None, filemode="a", use_color=True):
        if filename is not None:
            self.empty = False
            filename = Path(filename)
            if filename.is_dir():
                timestr = self._get_timestr().replace(" ", "_").replace(":", "-")
                filename = filename / f"log_{timestr}.log"
            self.file = open(filename, filemode)
        else:
            self.empty = True

        self.use_color = use_color

    def _get_timestr(self):
        n = datetime.now()
        return f"{n.year:04d}-{n.month:02d}-{n.day:02d} {n.hour:02d}:{n.minute:02d}:{n.second:02d}"

    def _write(self, msg, level):
        timestr = self._get_timestr()
        out = f"[{timestr} {level}] {msg}"

        if self.use_color:
            if level == " INFO":
                print("\033[34m" + out + "\033[0m")
            elif level == " WARN":
                print("\033[35m" + out + "\033[0m")
            elif level == "ERROR":
                print("\033[31m" + out + "\033[0m")
            elif level == "FATAL":
                print("\033[43m\033[1m" + out + "\033[0m")
            else:
                print(out)
        else:
            print(out)

        if not self.empty:
            self.file.write(out + "\r\n")

    def debug(self, *msg):
        msg = " ".join(map(str, msg))
        self._write(msg, "DEBUG")

    def info(self, *msg):
        msg = " ".join(map(str, msg))
        self._write(msg, " INFO")

    def warn(self, *msg):
        msg = " ".join(map(str, msg))
        self._write(msg, " WARN")

    def error(self, *msg):
        msg = " ".join(map(str, msg))
        self._write(msg, "ERROR")

    def fatal(self, *msg):
        msg = " ".join(map(str, msg))
        self._write(msg, "FATAL")

    def flush(self):
        if not self.empty:
            self.file.flush()
