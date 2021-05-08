from .common import *


class MyDatasetVer5(Dataset):
    """
    level에 따라서 json 형식의 {}, ""= 같은게 오히려 도움을 줄 수도 있기 때문에 최대한 원형을 유지하고, 진짜로 필요없다고 생각되는 정보들만 제거/치환한다.
    """

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
        text = text.replace("\r\n", "\n")
        text = text.replace("\\n", "\n")
        text = re.sub("(" + "|".join(seasons) + ")", " month ", text, flags=re.I)  # 월 치환
        text = re.sub(r"\d{2} \d{2}:\d{2}:\d{2}", " day time ", text)  # 일 + 시간 치환
        text = re.sub(r"\d{2}:\d{2}:\d{2}", " time ", text)  # 시간 치환
        text = re.sub(r"\d{2,4}-\d{1,2}-\d{1,2}", " date ", text)  # 년월일 치환
        text = re.sub(r"\[\d+\]", " pid ", text)  # PID 치환 - ssh[3256]
        text = re.sub(r"\(\d+\.\d+:\d+\)", " value ", text)  # value 치환
        text = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "ip", text)  # IP주소 치환
        text = re.sub(r"\s{2,}", " ", text)  # 중복되는 공백 치환
        text = text.strip()

        return text
