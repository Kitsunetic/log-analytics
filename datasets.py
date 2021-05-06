import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

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


class MyDataset(Dataset):
    def __init__(self, tokenizer, ids, texts, levels=None, ver=1) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.ids = ids
        self.texts = texts
        self.levels = levels
        self.train = levels is not None
        self.ver = ver

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        text = self.texts[idx]
        text = self.refine_data(text)
        if len(text) > 512:
            text = text[:512]
        org_text = text
        text = self.tokenizer.encode(text, add_special_tokens=True)
        if len(text) < 512:  # padding
            text = text + [0] * (512 - len(text))
        text = torch.tensor(text, dtype=torch.long)

        id = self.ids[idx]

        if self.train:  # train
            level = self.levels[idx]
            level = torch.tensor(level, dtype=torch.long)
            return id, text, level, org_text
        else:  # test
            return id, text, org_text

    def refine_data(self, full_log):
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

        if self.ver == 2:
            # ver2 2021-05-05 추가
            ############################################################################
            # IP주소 + port
            full_log = remove_pattern(r"\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}:\d+,?", full_log)
            full_log = remove_pattern(r"\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3},?", full_log)
            # port
            full_log = remove_pattern(r"port \d{1,5},?", full_log)
            # level
            full_log = remove_pattern(r"level\s*:\s*\d,?", full_log)
            # log
            full_log = remove_pattern(r"log\s*:", full_log)
            # uid, gid
            full_log = remove_pattern(r"[ug]id=\d+,?", full_log)
            # user
            full_log = remove_pattern(r"user=\w+", full_log)
            # logname=
            full_log = remove_pattern(r"logname=", full_log)
            ############################################################################

        full_log = full_log.replace("\n", " ")
        full_log = full_log.replace("\r", " ")
        full_log = re.sub(r"\s+", " ", full_log)

        return full_log


def load_train_data(data_dir, seed, fold, tokenizer, batch_size, num_workers, ver):
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "train.csv").to_numpy()
    ids = df[:, 0].astype(np.long)
    levels = df[:, 1].astype(np.long)
    texts = df[:, 2]

    skf = StratifiedKFold(shuffle=True, random_state=seed)
    indices = list(skf.split(texts, levels))
    tidx, vidx = indices[fold - 1]
    tds = MyDataset(tokenizer, ids[tidx], texts[tidx], levels[tidx], ver)
    vds = MyDataset(tokenizer, ids[vidx], texts[vidx], levels[vidx], ver)
    tdl = DataLoader(tds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    vdl = DataLoader(vds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    return tdl, vdl


def load_train_total_data(data_dir, tokenizer, batch_size, num_workers, ver):
    # 실험 용으로 validation 없이 모든 데이터에 대한 dataloader.
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "train.csv").to_numpy()
    ids = df[:, 0].astype(np.long)
    levels = df[:, 1].astype(np.long)
    texts = df[:, 2]

    ds = MyDataset(tokenizer, ids, texts, levels, ver=ver)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    return dl


def load_test_data(data_dir, seed, fold, tokenizer, batch_size, num_workers, ver):
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "test.csv").to_numpy()
    ids = df[:, 0].astype(np.long)
    texts = df[:, 1]

    ds = MyDataset(tokenizer, ids, texts, ver=ver)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    df = pd.read_csv(data_dir / "validation_sample.csv").to_numpy()
    ids = np.array([0, 1, 2], dtype=np.long)
    texts = df[:, 0]
    ds2 = MyDataset(tokenizer, ids, texts, ver=ver)
    dl2 = DataLoader(ds2, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return dl, dl2
