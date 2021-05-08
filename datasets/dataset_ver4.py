from .common import *


class MyDatasetVer4(Dataset):
    def __init__(self, tokenizer, ids, texts, levels=None, ver=4) -> None:
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
        otext = text  # backup original text

        text = self.tokenizer.encode(text, add_special_tokens=True)
        if len(text) < 512:  # padding
            text = text + [0] * (512 - len(text))
        elif len(text) > 512:
            text = text[:512]
        text = torch.tensor(text, dtype=torch.long)

        # data id
        id = self.ids[idx]

        if self.train:  # train
            level = self.levels[idx]
            level = torch.tensor(level, dtype=torch.long)
            return id, text, level, otext
        else:  # test
            return id, text, otext

    def refine_data(self, text):
        text = text.replace("\n", " { ")
        text = text.replace("\\n", " { ")
        text = re.sub(r"[\[\]\(\)\"'\s\|\+\*\&\^\%\$\#\@\!\~\`;\?\<\>]+", " ", text)
        # text = text.replace("=", " = ")
        text = text.replace("{", "[SEP]")
        text = text.replace("}", "[SEP]")
        # text = text.replace(".", ". ")

        # 날짜나 숫자로 시작하면 지우기
        text = text.split()
        while True:
            if text[0].isdigit() or text[0].lower() in seasons:
                text = text[1:]
            else:
                break

        text = " ".join(list(filter(None, map(self.myfilter, text))))

        return text

    def myfilter(self, x):
        # IP주소는 "IP" 로 치환
        if re.fullmatch(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", x):
            return "IP"

        return x
