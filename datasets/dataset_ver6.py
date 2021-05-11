import pickle

from .common import *


def refine_data(text):
    text = text.replace("\r\n", "\n")
    text = text.replace("\\n", "\n")
    text = text.replace("\\", "")
    # text = re.sub("(" + "|".join(seasons) + ")", " month ", text, flags=re.I)  # 월 치환
    # text = re.sub(r"\d{2} \d{2}:\d{2}:\d{2}", " day time ", text)  # 일 + 시간 치환
    # text = re.sub(r"\d{2}:\d{2}:\d{2}", " time ", text)  # 시간 치환
    # text = re.sub(r"\d{2,4}-\d{1,2}-\d{1,2}", " date ", text)  # 년월일 치환
    # text = re.sub(r"\[\d+\]", " pid ", text)  # PID 치환 - ssh[3256]
    # text = re.sub(r"\(\d+\.\d+:\d+\)", " value ", text)  # value 치환
    # text = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ip ", text)  # IP주소 치환
    # text = re.sub(r"port \d{1,5}", " port ", text, flags=re.I)  # IP주소 치환
    text = re.sub(r"\s{2,}", " ", text)  # 중복되는 공백 치환
    text = text.strip()

    return text


class MyDatasetVer6(Dataset):
    """
    level에 따라서 json 형식의 {}, ""= 같은게 오히려 도움을 줄 수도 있기 때문에 최대한 원형을 유지하고, 진짜로 필요없다고 생각되는 정보들만 제거/치환한다.
    """

    def __init__(self, tokenizer, texts, levels=None) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.texts = texts
        self.levels = levels
        self.train = levels is not None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        text = refine_data(str(text))
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


class MyDatasetVer6Test(Dataset):
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
        text = refine_data(text)
        otext = text  # backup original text before being tokenized

        text = self.tokenizer.encode(text, add_special_tokens=True)
        if len(text) < 512:  # padding
            text = text + [0] * (512 - len(text))
        elif len(text) > 512:  # cropping
            text = text[:512]
        text = torch.tensor(text, dtype=torch.long)

        return text, otext, ids


class DatasetGeneratorVer6:
    def __init__(
        self,
        data_dir,
        seed,
        fold,
        tokenizer,
        batch_size,
        num_workers,
        train_shuffle=True,
        oversampling=False,
        oversampling_scale=50,
    ):
        self.data_dir = Path(data_dir)
        self.seed = seed
        self.fold = fold
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_shuffle = train_shuffle
        self.oversampling = oversampling
        self.oversampling_scale = oversampling_scale

        self.dl_kwargs = dict(batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def train_valid(self):
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
        tidx, vidx = indices[self.fold]
        tds = MyDatasetVer6(self.tokenizer, texts[tidx], levels[tidx])
        vds = MyDatasetVer6(self.tokenizer, texts[vidx], levels[vidx])

        tdl = DataLoader(tds, shuffle=self.train_shuffle, **self.dl_kwargs)
        vdl = DataLoader(vds, shuffle=False, **self.dl_kwargs)
        return tdl, vdl

    def train_only(self):
        with open(self.data_dir / "train.pkl", "rb") as f:
            data = pickle.load(f)
        levels = np.array(data["level"], dtype=np.long)
        texts = np.array(data["text"], dtype=np.object)

        ds = MyDatasetVer6(self.tokenizer, texts, levels)
        dl = DataLoader(ds, shuffle=False, **self.dl_kwargs)
        return dl

    def valid_lv7(self):
        # validation level 7 dataset
        with open(self.data_dir / "valid-level7.pkl", "rb") as f:
            texts = pickle.load(f)

        ds = MyDatasetVer6(self.tokenizer, texts)
        dl = DataLoader(ds, shuffle=False, **self.dl_kwargs)
        return dl

    def test(self):
        # test dataset
        with open(self.data_dir / "test.pkl", "rb") as f:
            data = pickle.load(f)

        ds = MyDatasetVer6Test(self.tokenizer, data)
        return ds
