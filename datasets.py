import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.core.numeric import full
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_optimizer
from pytorch_transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)
from sklearn.model_selection import StratifiedKFold
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import FocalLoss, AverageMeter
import re

seasons = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def first_word(text, deli=" "):
    for i, t in enumerate(text):
        if t == deli:
            break
    return text[:i]


def remove_pattern(pattern, full_log):
    for s in re.finditer(pattern, full_log):
        a, b = s.span()
        full_log = (full_log[:a] + full_log[b:]).strip()
    return full_log


def remove_date(full_log):
    t = first_word(full_log)
    if len(t) == 4 and t.isdigit() and t[:2] in ("19", "20", "21"):
        full_log = full_log[5:].strip()

    t = first_word(full_log)
    if len(t) == 3 and t in seasons:
        full_log = full_log[4:].strip()

        t = first_word(full_log)
        if t.isdigit():
            full_log = full_log[len(t) + 1 :].strip()

    # 00:00:00 형식의 시간 이면?
    if re.match(r"\d{2}:\d{2}:\d{2}", full_log):
        full_log = full_log[9:].strip()

    if full_log.startswith("localhost"):
        full_log = full_log[10:].strip()

    # sshd[pid] 에서 pid 제거
    # t = first_word(full_log)
    # if re.match(r"[\w\d]+\[\d+\]", t):
    #    u = first_word(t, deli="[")
    #    full_log = (u + " " + full_log[len(t) + 1 :]).strip()

    # @timestamp: "~~~~Z"
    full_log = remove_pattern(r'"@timestamp"\s?:\s?"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?",?', full_log)
    # "pid": "4567"
    full_log = remove_pattern(r'"pid"\s?:\s?\d+,?', full_log)
    # [pid]
    full_log = remove_pattern(r"\[\d+\]", full_log)

    full_log = full_log.replace("\n", " ")
    full_log = full_log.replace("\r", " ")
    full_log = re.sub(r"\s+", " ", full_log)

    return full_log


class MyDataset(Dataset):
    def __init__(self, tokenizer, ids, texts, levels=None) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.ids = ids
        self.texts = texts
        self.levels = levels
        self.train = levels is not None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        text = self.texts[idx]
        text = remove_date(text)
        if len(text) > 512:
            text = text[:512]
        text = self.tokenizer.encode(text, add_special_tokens=True)
        if len(text) < 512:  # padding
            text = text + [0] * (512 - len(text))
        text = torch.tensor(text, dtype=torch.long)

        id = self.ids[idx]

        if self.train:  # train
            level = self.levels[idx]
            level = torch.tensor(level, dtype=torch.long)
            return id, text, level
        else:  # test
            return id, text


def load_train_data(data_dir, seed, fold, tokenizer, batch_size, num_workers):
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "train.csv").to_numpy()
    ids = df[:, 0].astype(np.long)
    levels = df[:, 1].astype(np.long)
    texts = df[:, 2]

    skf = StratifiedKFold(shuffle=True, random_state=seed)
    indices = list(skf.split(texts, levels))
    tidx, vidx = indices[fold - 1]
    tds = MyDataset(tokenizer, ids[tidx], texts[tidx], levels[tidx])
    vds = MyDataset(tokenizer, ids[vidx], texts[vidx], levels[vidx])
    tdl = DataLoader(tds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    vdl = DataLoader(vds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    return tdl, vdl
