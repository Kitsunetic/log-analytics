from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from .dataset_ver1 import MyDatasetVer1
from .dataset_ver2 import MyDatasetVer2
from .dataset_ver3 import MyDatasetVer3
from .dataset_ver4 import MyDatasetVer4
from .dataset_ver5 import MyDatasetVer5


def get_dataset_class(ver):
    D = {
        1: MyDatasetVer1,
        2: MyDatasetVer2,
        3: MyDatasetVer3,
        4: MyDatasetVer4,
        5: MyDatasetVer5,
    }
    if ver not in D:
        raise NotImplementedError(f"Unknown dataset version {ver}")

    return D[ver]


def load_train_data(data_dir, seed, fold, tokenizer, batch_size, num_workers, ver, train_shuffle=True):
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "train.csv").to_numpy()
    ids = df[:, 0].astype(np.long)
    levels = df[:, 1].astype(np.long)
    texts = df[:, 2]

    skf = StratifiedKFold(shuffle=True, random_state=seed)
    indices = list(skf.split(texts, levels))
    tidx, vidx = indices[fold - 1]
    DatasetClass = get_dataset_class(ver)
    tds = DatasetClass(tokenizer, ids[tidx], texts[tidx], levels[tidx], ver)
    vds = DatasetClass(tokenizer, ids[vidx], texts[vidx], levels[vidx], ver)
    tdl = DataLoader(tds, batch_size=batch_size, shuffle=train_shuffle, pin_memory=True, num_workers=num_workers)
    vdl = DataLoader(vds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    return tdl, vdl


def load_train_total_data(data_dir, tokenizer, batch_size, num_workers, ver):
    # 실험 용으로 validation 없이 모든 데이터에 대한 dataloader.
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "train.csv").to_numpy()
    ids = df[:, 0].astype(np.long)
    levels = df[:, 1].astype(np.long)
    texts = df[:, 2]

    DatasetClass = get_dataset_class(ver)
    ds = DatasetClass(tokenizer, ids, texts, levels, ver=ver)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    return dl


def load_test_data(data_dir, seed, fold, tokenizer, batch_size, num_workers, ver):
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "test.csv").to_numpy()
    ids = df[:, 0].astype(np.long)
    texts = df[:, 1]

    DatasetClass = get_dataset_class(ver)
    ds = DatasetClass(tokenizer, ids, texts, ver=ver)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    df = pd.read_csv(data_dir / "validation_sample.csv").to_numpy()
    ids = np.array([0, 1, 2], dtype=np.long)
    texts = df[:, 0]
    ds2 = DatasetClass(tokenizer, ids, texts, ver=ver)
    dl2 = DataLoader(ds2, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return dl, dl2
