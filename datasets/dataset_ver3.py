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


class MyDatasetVer3(Dataset):
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

    def filttt(self, x):
        if len(x) == 1:
            return False

        if re.fullmatch(r"[\.\d,\s-]+([ABTZ]|ms)?", x):
            return False

        if x.lower() in ("x64", "win32", "x86", "ko", "en", "kr", "us", "ko-kr", "en-us"):
            return False

        return True

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

        # ver3 2021-05-06 추가 - 숫자/특수문자 제거
        ############################################################################
        # IP주소 + port
        full_log = remove_pattern(r"\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}:\d+,?", full_log)
        full_log = remove_pattern(r"\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3},?", full_log)
        # port
        full_log = remove_pattern(r"port \d{1,5},?", full_log)

        full_log = re.sub(r"[\[\]\{\}\(\):\"'\s=\|\+\-\_\*\&\^\%\$\#\@\!\~\`,;\?]+", " ", full_log)
        full_log = " ".join(filter(self.filttt, full_log.split()))
        ############################################################################

        full_log = re.sub(r"\s+", " ", full_log)

        return full_log
