import argparse
import math
import multiprocessing
import random
import sys
from datetime import datetime
from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_optimizer
import yaml
from easydict import EasyDict
from pytorch_transformers import (
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
from transformers import (
    AlbertForSequenceClassification,
    AlbertTokenizer,
    DebertaForSequenceClassification,
    DebertaTokenizer,
    SqueezeBertForSequenceClassification,
    SqueezeBertTokenizer,
    XLNetForSequenceClassification,
    XLNetTokenizer,
)

from datasets import load_test_data, load_train_data, load_train_total_data
from main import MyTrainer
from utils import SAM, AverageMeter, CustomLogger, FocalLoss, seed_everything


activation = []


def hook(model, input, output):
    activation.append(output.detach().cpu())


def get_dist(deck, feat, topk):
    dist = torch.norm(deck - feat[None], dim=1, p=None)
    values, indices = dist.topk(topk, largest=False)  # knn
    return values, indices


def step1(C, fold):
    global activation

    trainer = MyTrainer(C, 1, C.result_dir / f"{C.uid}_{fold}.pth")
    model = trainer.model
    model.eval()
    torch.set_grad_enabled(False)
    model.pre_classifier.register_forward_hook(hook)

    dl = load_train_total_data(C.dataset.dir, trainer.tokenizer, 100, 6, ver=C.dataset.ver)

    activation = []
    deck = {
        "fcfeat": [],
        "tlevel": [],
        "fclevel": [],
        "otext": [],
    }
    with tqdm(total=len(dl.dataset), ncols=100, file=sys.stdout) as t:
        for id, text, tlevel, otext in dl:
            pred = model(text.cuda(non_blocking=True))[0].cpu()
            deck["fcfeat"].append(pred)
            deck["tlevel"].append(tlevel)
            deck["fclevel"].append(pred.argmax(dim=1))
            deck["otext"].extend(otext)

            t.update(len(id))

    deck["fcfeat"] = torch.cat(deck["fcfeat"])
    deck["tlevel"] = torch.cat(deck["tlevel"])
    deck["fclevel"] = torch.cat(deck["fclevel"])
    deck["feat"] = torch.cat(activation)

    torch.save(
        dict(
            fcfeat=deck["fcfeat"].numpy(),
            tlevel=deck["tlevel"].numpy(),
            fclevel=deck["fclevel"].numpy(),
            feat=deck["feat"].numpy(),
            otext=deck["otext"],
        ),
        C.result_dir / f"{C.uid}_{fold}-deck1.pth",
    )


def step2(C, fold):
    global activation

    trainer = MyTrainer(C, 1, C.result_dir / f"{C.uid}_{fold}.pth")
    model = trainer.model
    model.eval()
    torch.set_grad_enabled(False)
    model.pre_classifier.register_forward_hook(hook)

    # sfeats 저장
    activation = []
    deck = {"fcfeat": [], "fclevel": [], "otext": []}
    with tqdm(total=len(trainer.dl_test.dataset), ncols=100, file=sys.stdout) as t:
        for _, text, otext in trainer.dl_test:
            pred = model(text.cuda(non_blocking=True))[0].cpu()
            deck["fcfeat"].append(pred)
            deck["fclevel"].append(pred.argmax(dim=1))
            deck["otext"].extend(otext)
            t.update(len(text))

    deck["fcfeat"] = torch.cat(deck["fcfeat"])
    deck["fclevel"] = torch.cat(deck["fclevel"])
    deck["feat"] = torch.cat(activation)

    torch.save(
        dict(
            fcfeat=deck["fcfeat"].numpy(),
            fclevel=deck["fclevel"].numpy(),
            feat=deck["feat"].numpy(),
            otext=deck["otext"],
        ),
        C.result_dir / f"{C.uid}_{fold}-deck2.pth",
    )


def step3(C, fold):
    # dist를 구함
    deck1 = torch.load(C.result_dir / f"{C.uid}_{fold}-deck1.pth")
    deck2 = torch.load(C.result_dir / f"{C.uid}_{fold}-deck2.pth")
    tdeck = {"feat": deck1["feat"].cuda(), "tlevel": deck1["tlevel"]}
    sdeck = {
        "feat": deck2["feat"].cuda(),
        "fcfeat": deck2["fcfeat"],
        "fclevel": deck2["fclevel"],
    }

    # dist를 구함
    dists, indices, fcfeats, tlevels = [], [], [], []
    with tqdm(total=len(sdeck["feat"]), ncols=100, file=sys.stdout) as t:
        for i in range(len(sdeck["feat"])):
            dist_, index_ = get_dist(tdeck["feat"], sdeck["feat"][i], 8)
            dist = dist_.cpu()
            index = index_.cpu()
            fcfeat = sdeck["fcfeat"][i]
            tlevel = tdeck["tlevel"][index]
            dists.append(dist)
            indices.append(index)
            fcfeats.append(fcfeat)
            tlevels.append(tlevel)

            t.update()

    dists_ = torch.stack(dists)
    indices_ = torch.stack(indices)
    fcfeats_ = torch.stack(fcfeats)
    tlevels_ = torch.stack(tlevels)

    torch.load(
        dict(
            dists=dists_.numpy(),
            indices=indices_.numpy(),
            fcfeats=fcfeats_.numpy(),
            tlevels=tlevels_.numpy(),
        ),
        C.result_dir / f"{C.uid}_{fold}-deck3.pth",
    )


# ver4
def politic_draw(dists, indices, fclevel, tlevels):
    dd = dists[-1] - dists[0]
    dist = dists[0]
    same = (tlevels == tlevels[0]).sum() == 8

    # policy1: dist와 관계 없이 앞의 5개가 모두 tlevels가 3 또는 5이면 그 값을 출력
    if tlevels[0] in [3, 5] and (tlevels[:5] == tlevels[0]).sum() == 5:
        return tlevels[0].item()

    # policy2: dist와 관계 없이 모든 tlevels 중 앞의 3개가 2 또는 4 또는 6이면 그 값을 출력
    if tlevels[0] in [2, 4, 6] and (tlevels[:3] == tlevels[0]).sum() == 3:
        return tlevels[0].item()

    # policy: dist가 0.5보다 크면 level 7
    if dist > 0.4:
        return 7

    # 나머지
    return tlevels[0].item()


def step4(C, fold):
    deck3 = torch.load(C.result_dir / f"{C.uid}_{fold}-deck3.pth")
    deck3 = {k: torch.from_numpy(v) for k, v in deck3.items()}
    deck3["fclevels"] = deck3["fcfeats"].argmax(1)

    N = len(deck3["dists"])
    outdic = {"id": list(range(1000000, 2418915 + 1)), "level": []}
    with tqdm(total=N, ncols=100, file=sys.stdout) as t:
        for i in range(N):
            v = politic_draw(deck3["dists"][i], deck3["indices"][i], deck3["fclevels"][i], deck3["tlevels"][i])
            outdic["level"].append(v)
            t.update()

    outdf = pd.DataFrame(outdic)
    outdf.to_csv(C.result_dir / f"{C.uid}_{fold}-ver4.csv", index=False)


steps = {
    1: step1,
    2: step2,
    3: step3,
    4: step4,
}


def main():
    args = argparse.ArgumentParser()
    args.add_argument("config", type=str)
    args.add_argument("fold", type=int)
    args.add_argument("step", type=int)
    args = args.parse_args(sys.argv[1:])

    with open(args.config, "r") as f:
        C = EasyDict(yaml.load(f, yaml.FullLoader))
        if C.dataset.num_workers < 0:
            C.dataset.num_workers = multiprocessing.cpu_count()
        C.uid = f"{C.model.name.split('/')[-1]}-{C.train.loss.name}"
        C.uid += f"-{C.train.optimizer.name}"
        C.uid += f"-lr{C.train.lr}"
        C.uid += f"-dsver{C.dataset.ver}"
        C.uid += "-sam" if C.train.SAM else ""
        C.uid += f"-{C.comment}" if C.comment is not None else ""
        print(C.uid)

        C.log = CustomLogger()
        C.result_dir = Path(C.result_dir)
        C.dataset.dir = Path(C.dataset.dir)
        seed_everything(C.seed, deterministic=False)

    step_fn = steps[args.step]
    fold = args.fold
    step_fn(C, fold)


if __name__ == "__main__":
    main()
