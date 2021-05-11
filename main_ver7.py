import argparse
import math
import multiprocessing
import sys
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
from pytorch_transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from sklearn.metrics import classification_report, f1_score
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from datasets.dataset_ver7 import DatasetGeneratorVer7
from utils import SAM, ArcFaceLoss, AverageMeter, CustomLogger, FocalLoss, seed_everything


class MyOutput:
    def __init__(self) -> None:
        self.loss = AverageMeter()
        self.acc = AverageMeter()
        self.plevels = []
        self.tlevels = []

    def freeze(self):
        self.loss = self.loss()
        self.acc = self.acc()
        self.plevels = torch.cat(self.plevels)
        self.tlevels = torch.cat(self.tlevels)
        return self


class MyTrainer:
    _tqdm_ = dict(ncols=100, leave=False, file=sys.stdout)

    def __init__(self, config, fold, checkpoint=None) -> None:
        self.C = config
        self.fold = fold

        # model
        if self.C.model.name.startswith("distilbert-"):
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.C.model.name)
            self.model = DistilBertForSequenceClassification.from_pretrained(self.C.model.name, num_labels=7).cuda()
        else:
            raise NotImplementedError(self.C.model.name)
        # loss
        if self.C.train.loss.name == "ce":
            self.criterion = nn.CrossEntropyLoss().cuda()
        elif self.C.train.loss.name == "focal":
            if "params" in self.C.train.loss:
                self.criterion = FocalLoss(self.C.train.loss.params.gamma).cuda()
            else:
                self.criterion = FocalLoss(self.C.train.loss.gamma).cuda()
        elif self.C.train.loss.name == "arcface":
            self.criterion = ArcFaceLoss(**self.C.train.loss.params)
        else:
            raise NotImplementedError(self.C.train.loss.name)
        # optimizer
        if self.C.train.optimizer.name == "Adam":
            self.optimizer = Adam(self.model.parameters(), lr=self.C.train.lr)
        elif self.C.train.optimizer.name == "AdamW":
            self.optimizer = AdamW(self.model.parameters(), lr=self.C.train.lr)
        else:
            raise NotImplementedError(self.C.train.optimizer.name)
        # scheduler
        if self.C.train.scheduler.name == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer, **self.C.train.scheduler.params)
        else:
            raise NotImplementedError(self.C.train.scheduler.name)

        self.epoch = 1
        self.best_loss = math.inf
        self.best_acc = 0.0
        self.earlystop_cnt = 0
        self._freeze_step = 3

        if checkpoint is not None:
            if Path(checkpoint).exists():
                self.load(checkpoint)
            else:
                self.C.log.info("No checkpoint file", checkpoint)

        # dataset
        self.dsgen = DatasetGeneratorVer7(
            self.C.dataset.dir,
            self.C.seed,
            self.fold,
            self.tokenizer,
            self.C.dataset.batch_size,
            self.C.dataset.num_workers,
            oversampling=self.C.dataset.oversampling,
            oversampling_scale=self.C.dataset.oversampling_scale,
        )
        self.tdl, self.vdl = self.dsgen.train_valid()

    def _freeze_step1(self):
        self._freeze_step = 1
        self.model.requires_grad_(False)
        self.model.classifier.requires_grad_(True)

    def _freeze_step2(self):
        self._freeze_step = 2
        self.model.requires_grad_(True)
        self.model.classifier.requires_grad_(False)

    def _freeze_step3(self):
        self._freeze_step = 3
        self.model.requires_grad_(True)

    def save(self, path):
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "best_loss": self.best_loss,
                "best_acc": self.best_acc,
                "earlystop_cnt": self.earlystop_cnt,
            },
            path,
        )

    def load(self, path):
        print("Load pretrained", path)
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epoch = ckpt["epoch"] + 1
        self.best_loss = ckpt["best_loss"]
        self.best_acc = ckpt["best_acc"]
        self.earlystop_cnt = ckpt["earlystop_cnt"]

    def train_loop(self):
        self.model.train()

        O = MyOutput()
        with tqdm(total=len(self.tdl.dataset), desc=f"Train {self.epoch:03d}", **self._tqdm_) as t:
            for text, tlevel, otext in self.tdl:
                text_ = text.cuda()
                tlevel_ = tlevel.cuda()
                plevel_ = self.model(text_)[0]
                loss = self.criterion(plevel_, tlevel_)

                self.optimizer.zero_grad()
                loss.backward()
                if isinstance(self.optimizer, SAM):
                    self.optimizer.first_step(zero_grad=True)
                    self.criterion(self.model(text_), tlevel_).backward()
                    self.optimizer.second_step()
                else:
                    self.optimizer.step()

                pvlevel_ = plevel_.detach().argmax(dim=1)
                acc = (pvlevel_ == tlevel_).sum() / len(text) * 100
                O.loss.update(loss.item(), len(text))
                O.acc.update(acc.item(), len(text))
                O.plevels.append(pvlevel_.cpu())
                O.tlevels.append(tlevel)
                t.set_postfix_str(f"loss: {O.loss():.6f}, acc: {O.acc():.2f}", refresh=False)
                t.update(len(text))
        return O.freeze()

    @torch.no_grad()
    def valid_loop(self):
        self.model.eval()

        O = MyOutput()
        with tqdm(total=len(self.vdl.dataset), desc=f"Valid {self.epoch:03d}", **self._tqdm_) as t:
            for text, tlevel, otext in self.vdl:
                text_ = text.cuda()
                tlevel_ = tlevel.cuda()
                plevel_ = self.model(text_)[0]
                loss = self.criterion(plevel_, tlevel_)

                pvlevel_ = plevel_.detach().argmax(dim=1)
                acc = (pvlevel_ == tlevel_).sum() / len(text) * 100
                O.loss.update(loss.item(), len(text))
                O.acc.update(acc.item(), len(text))
                O.plevels.append(pvlevel_.cpu())
                O.tlevels.append(tlevel)
                t.set_postfix_str(f"loss: {O.loss():.6f}, acc: {O.acc():.2f}", refresh=False)
                t.update(len(text))
        return O.freeze()

    @torch.no_grad()
    def callback(self, to: MyOutput, vo: MyOutput):
        # f1 score
        tf1 = f1_score(to.tlevels, to.plevels, zero_division=1, average="macro")
        vf1 = f1_score(vo.tlevels, vo.plevels, zero_division=1, average="macro")
        trep = str(classification_report(to.tlevels, to.plevels, labels=[0, 1, 2, 3, 4, 5, 6], zero_division=1))
        vrep = str(classification_report(vo.tlevels, vo.plevels, labels=[0, 1, 2, 3, 4, 5, 6], zero_division=1))

        self.C.log.info(
            f"Epoch: {self.epoch:03d}/{self.C.train.max_epochs},",
            f"loss: {to.loss:.6f};{vo.loss:.6f},",
            f"acc {to.acc:.2f};{vo.acc:.2f}",
            f"f1 {tf1:.2f}:{vf1:.2f}",
        )
        self.C.log.info("Train Report\r\n" + trep)
        self.C.log.info("Validation Report\r\n" + vrep)
        self.C.log.flush()

        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(vo.loss)

        if self.best_loss - vo.loss > 1e-6 or vf1 - self.best_acc > 1e-6:
            if self.best_loss > vo.loss:
                self.best_loss = vo.loss
            else:
                self.best_acc = vf1

            self.earlystop_cnt = 0
            self.save(self.C.result_dir / f"{self.C.uid}_{self.fold}.pth")

            # TODO 결과 요약 이미지 출력
        else:
            self.earlystop_cnt += 1

    def fit(self):
        for self.epoch in range(self.epoch, self.C.train.max_epochs + 1):
            if self.C.train.finetune.do:
                if self.epoch <= self.C.train.finetune.step1_epochs:
                    if self._freeze_step != 1:
                        self.C.log.info("Finetune Step 1")
                        self._freeze_step1()
                elif self.epoch <= self.C.train.finetune.step2_epochs:
                    if self._freeze_step != 2:
                        self.C.log.info("Finetune Step 2")
                        self._freeze_step2()
                elif self.epoch > self.C.train.finetune.step2_epochs:
                    if self._freeze_step != 3:
                        self.C.log.info("Finetune Step 3")
                        self._freeze_step3()

            to = self.train_loop()
            vo = self.valid_loop()
            self.callback(to, vo)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("config", type=str)
    args = args.parse_args(sys.argv[1:])

    with open(args.config, "r") as f:
        C = EasyDict(yaml.load(f, yaml.FullLoader))

    for fold, checkpoint in zip(C.train.folds, C.train.checkpoints):
        with open(args.config, "r") as f:
            C = EasyDict(yaml.load(f, yaml.FullLoader))
            Path(C.result_dir).mkdir(parents=True, exist_ok=True)

            if C.dataset.num_workers < 0:
                C.dataset.num_workers = multiprocessing.cpu_count()
            C.uid = f"{C.model.name.split('/')[-1]}-{C.train.loss.name}"
            C.uid += f"-{C.train.optimizer.name}"
            C.uid += f"-lr{C.train.lr}"
            C.uid += f"-ver{C.ver}" if C.ver > 1 else ""
            C.uid += f"-os{C.dataset.oversampling_scale}" if C.dataset.oversampling else ""
            C.uid += "-sam" if C.train.SAM else ""
            C.uid += f"-{C.comment}" if C.comment is not None else ""
            print(C.uid)

            log = CustomLogger(Path(C.result_dir) / f"{C.uid}_{fold}.log", "a")
            log.info("\r\n" + pformat(C))
            log.flush()

            C.log = log
            C.result_dir = Path(C.result_dir)
            C.dataset.dir = Path(C.dataset.dir)
            seed_everything(C.seed, deterministic=False)

        C.log.info("Fold", fold, ", checkpoint", checkpoint)
        trainer = MyTrainer(C, fold, checkpoint)
        trainer.fit()


if __name__ == "__main__":
    main()
