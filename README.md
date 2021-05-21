# 로그 분석을 통한 보안 위험도 예측 AI 경진대회

**Public 0.92089, Private 0.92294, 최종 2/152위**

# 전체 요약

## 1. 전처리

**중복되는 데이터 제거**

* 똑같은 텍스트를 갖은 데이터가 여럿 있습니다. 이 데이터들을 제거하면 train과 inference 시간이 약 2/3로 개선되었습니다.
* 또한 텍스트는 똑같지만 level은 다른 데이터도 몇 있습니다. 중복되는 데이터 중에서 가장 많이 등장한 level만 남기고 전부 제거했습니다.

**필요없는 텍스트 제거**

* 날짜, 시간, PID, timestamp 등 필요없다고 생각되는 부분을 제거했습니다.

**Oversampling**

* level 2, 4, 6의 개수가 너무 적습니다. Train-validation을 나누고 나면 일부 특징은 해당 fold에서 아예 누락되어 학습을 못하기도 하기 때문에 level 2, 4, 6 인 데이터만 10배로 oversampling 해주었습니다. Oversampling할 때 특별히 augmenation을 적용하지는 않았고 그냥 복제를 했습니다.

**기타**

* token의 길이가 512를 초과하는 경우 앞의 512자리까지만 사용했습니다.
* 원본 데이터 파일은 `./data/ori` 에 저장됩니다.
* 데이터를 전처리해서 `pkl`파일으로 `./data/ver6` 에 저장했습니다.

## 2. 학습

* DistilBert를 finetune했고, FocalLoss를 썼습니다. 추가 데이터는 없습니다.
* 5 fold cross validation을 했습니다. 하지만 결과를 합치기 전에 public score 0.9207, 합친 후에 0.9208으로 크게 차이는 없었습니다.

## 3. 후처리

* 모든 train 데이터의 feature를 저장해두고 test feature와 euclidean distance를 계산합니다. 이 거리 값을 level 추론에 사용합니다.

## 4. 추론

* "기존에 나타난 log들과는 상이한 데이터가 level7지 않을까?" 하는 가정으로 접근했습니다.
* Level에 따라서 threshold를 각각 설정해주었고, threshold 또는 각종 조건을 넘어서면 level7, 아닐경우 fully connected layer의 출력과 distance를 종합해서 출력을 만들었습니다.

## 사전학습 가중치

https://drive.google.com/drive/u/3/folders/1XD83QcktSV-YaRJ618llW1tUjQDaBo3r

## Environments

* Ubuntu 18.04 LTS
* RTX3090, cuda-toolkit 11.2
* Checkpoint를 통해서 여러 장비들을 계속 옮겨가면서 작업했기 때문에 random seed 등의 문제로 완전한 reproduce는 어려울 수 있습니다.

## Requirements

* pytorch==1.7.1 # 이유는 모르겠지만 같은 weight를 써도 transformer 계열의 모델들은 1.7.1과 1.8.0 버전에서 전혀 다른 출력을 내더군요
* numpy
* pandas
* matplotlib
* pyaml
* easydict
* pytorch_transformers
* sklearn
* tqdm

# 실행 방법

## 1. 학습

```bash
python main.py config/distilbert-base-uncased-ver7.yaml
```

## 2. Deck 생성

`scripts/01. deck만들기.ipynb`를 따라합니다.  
Cross validation을 위해서는 5개의 fold마다 각각 총 5회 반복합니다.

## 3. Distance 생성

`scripts/02. dist구하기.ipynb`를 따라합니다.  
Cross validation을 위해서는 5개의 fold마다 각각 총 5회 반복합니다.

## 4. Submission 파일 생성

`scripts/03. submission 만들기.ipynb`를 따라합니다.  
Cross validation을 위해서는 5개의 fold마다 각각 총 5회 반복합니다.

## 5. Ensemble

`scripts/04. ensemble.ipynb`를 따라합니다.
