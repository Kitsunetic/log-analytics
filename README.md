# 로그 분석을 통한 보안 위험도 예측 AI 경진대회

**Public 0.92089, Private 0.92294, 최종 2/152위**

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
