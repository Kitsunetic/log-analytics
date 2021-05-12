import pickle
import random

from .common import *


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


def filttt(x):
    if len(x) == 1:
        return False

    if re.fullmatch(r"[\.\d,\s-]+([ABTZ]|ms)?", x):
        return False

    if x.lower() in ("x64", "win32", "x86", "ko", "en", "kr", "us", "ko-kr", "en-us"):
        return False

    return True


def split_text(text):
    se = re.finditer(r"[\s\:\=\,\.\{\}\(\)\[\]]+", text)
    i = 0
    ps = []
    for s in se:
        a, b = s.span()
        ps.append((i, a))
        i = b

    return ps


def EDA_RD(text, p=0.5, q=0.05):
    ps = split_text(text)
    out = ""
    i = 0
    for a, b in ps:
        out += text[i:a]
        if random.random() > q:
            out += text[a:b]
        i = b

    return out


def refine_data(full_log, augmentation=False):
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

    # @timestamp: "~~~~Z"
    full_log = remove_pattern(r'"@timestamp"\s?:\s?"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?",?', full_log)
    # "pid": "4567"
    full_log = remove_pattern(r'"pid"\s?:\s?\d+,?', full_log)
    # [pid]
    full_log = remove_pattern(r"\[\d+\]", full_log)

    # ver8: augmentation
    full_log = EDA_RD(full_log, p=0.5, q=0.05)

    # duplicate spaces
    full_log = re.sub(r"\s+", " ", full_log).strip()

    return full_log


class MyDatasetVer8(Dataset):
    """
    가장 결과가 좋았던 ver1을 따라간다
    Augmentation 추가
    """

    def __init__(self, tokenizer, texts, levels=None, augmentation=False) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.texts = texts
        self.levels = levels
        self.train = levels is not None
        self.augmentation = augmentation

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        text = refine_data(str(text), self.augmentation)
        otext = text  # backup original text before being tokenized

        text = self.tokenizer.encode(text, add_special_tokens=True)
        if len(text) < 512:  # padding
            text = text + [0] * (512 - len(text))
        elif len(text) > 512:  # cropping
            text = text[:512]
        text = torch.tensor(text, dtype=torch.long)

        if self.train:  # train
            level = self.levels[idx]
            level = torch.tensor(level, dtype=torch.long)
            return text, level, otext
        else:  # test
            return text, otext


class MyDatasetVer8Test(Dataset):
    def __init__(self, tokenizer, data) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data = data
        self.keys = list(self.data.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        text = self.keys[idx]
        ids = self.data[text]
        text = refine_data(text, False)
        otext = text  # backup original text before being tokenized

        text = self.tokenizer.encode(text, add_special_tokens=True)
        if len(text) < 512:  # padding
            text = text + [0] * (512 - len(text))
        elif len(text) > 512:  # cropping
            text = text[:512]
        text = torch.tensor(text, dtype=torch.long)

        return text, otext, ids


class DatasetGeneratorVer8:
    def __init__(self, C, tokenizer, shuffle=True):
        self.data_dir = Path(C.dataset.dir)
        self.seed = C.seed
        self.tokenizer = tokenizer
        self.batch_size = C.dataset.batch_size
        self.num_workers = C.dataset.num_workers
        self.train_shuffle = shuffle
        self.oversampling = C.dataset.oversampling
        self.oversampling_scale = C.dataset.oversampling_scale

        self.dl_kwargs = dict(batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def train_valid(self, fold):
        # train dataset
        with open(self.data_dir / "train.pkl", "rb") as f:
            data = pickle.load(f)
        levels = np.array(data["level"], dtype=np.long)
        texts = np.array(data["text"], dtype=np.object)

        if self.oversampling:
            # oversampling -- level 2
            mask = levels == 2
            o2_len = len(levels[mask])
            o2_levels = np.array([2] * o2_len * self.oversampling_scale)
            o2_texts = np.concatenate([texts[mask] for _ in range(self.oversampling_scale)])

            # oversampling -- level 4
            mask = levels == 4
            o4_len = len(levels[mask])
            o4_levels = np.array([4] * o4_len * self.oversampling_scale)
            o4_texts = np.concatenate([texts[mask] for _ in range(self.oversampling_scale)])

            # oversampling -- level 6
            mask = levels == 6
            o6_len = len(levels[mask])
            o6_levels = np.array([6] * o6_len * self.oversampling_scale)
            o6_texts = np.concatenate([texts[mask] for _ in range(self.oversampling_scale)])

            levels = np.concatenate([levels, o2_levels, o4_levels, o6_levels])
            texts = np.concatenate([texts, o2_texts, o4_texts, o6_texts])

        # k-fold
        skf = StratifiedKFold(n_splits=5, shuffle=self.train_shuffle, random_state=self.seed)
        indices = list(skf.split(texts, levels))
        tidx, vidx = indices[fold - 1]
        tds = MyDatasetVer8(self.tokenizer, texts[tidx], levels[tidx], augmentation=True)
        vds = MyDatasetVer8(self.tokenizer, texts[vidx], levels[vidx], augmentation=False)

        tdl = DataLoader(tds, shuffle=self.train_shuffle, **self.dl_kwargs)
        vdl = DataLoader(vds, shuffle=False, **self.dl_kwargs)
        return tdl, vdl

    def train_only(self):
        with open(self.data_dir / "train.pkl", "rb") as f:
            data = pickle.load(f)
        levels = np.array(data["level"], dtype=np.long)
        texts = np.array(data["text"], dtype=np.object)

        ds = MyDatasetVer8(self.tokenizer, texts, levels, augmentation=False)
        dl = DataLoader(ds, shuffle=False, **self.dl_kwargs)
        return dl

    def valid_lv7(self):
        # validation level 7 dataset
        with open(self.data_dir / "valid-level7.pkl", "rb") as f:
            texts = pickle.load(f)

        ds = MyDatasetVer8(self.tokenizer, texts, augmentation=False)
        dl = DataLoader(ds, shuffle=False, **self.dl_kwargs)
        return dl

    def test(self):
        # test dataset
        with open(self.data_dir / "test.pkl", "rb") as f:
            data = pickle.load(f)

        ds = MyDatasetVer8Test(self.tokenizer, data)
        return ds
