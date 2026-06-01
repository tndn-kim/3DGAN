"""
classify.py
-----------
분류 모델 학습 및 평가 모듈.
RandomForest / XGBoost / BiGRU 세 모델을 동일한 데이터로 비교.

원본 코드 수정사항:
  1. feature_size / y 미정의 변수 수정
  2. BiGRU 입력 shape 수정: [B, 76] → unsqueeze(1) → [B, 1, 76]
  3. 데이터 로딩 하드코딩 제거 → pipeline.py 에서 전달
  4. 클래스별 샘플 수 하드코딩 제거 → config 로 일반화
  5. 공통 지표 계산 함수로 중복 코드 통합
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import MinMaxScaler

import xgboost as xgb


# ────────────────────────────────────────────────
# 공통 지표 계산
# ────────────────────────────────────────────────

def _compute_metrics(model_name: str,
                     y_true: np.ndarray,
                     y_pred: np.ndarray) -> dict:
    """
    Confusion Matrix 기반 지표 계산 및 출력.
    FPR(오탐) / FNR(미탐) / F1 / Accuracy 포함.

    Returns:
        {
            'model'     : str,
            'accuracy'  : float,
            'f1_macro'  : float,
            'micro_FPR' : float,
            'micro_FNR' : float,
            'macro_FPR' : float,
            'macro_FNR' : float,
            'conf_matrix': np.ndarray,
            'class_FPR' : list,
            'class_FNR' : list,
        }
    """
    conf  = confusion_matrix(y_true, y_pred)
    n_cls = conf.shape[0]

    class_FPR, class_FNR = [], []
    for i in range(n_cls):
        TP = conf[i, i]
        FN = conf[i, :].sum() - TP
        FP = conf[:, i].sum() - TP
        TN = conf.sum() - (TP + FP + FN)
        class_FPR.append(FP / (FP + TN) if (FP + TN) > 0 else 0.0)
        class_FNR.append(FN / (FN + TP) if (FN + TP) > 0 else 0.0)

    FP_total = sum(conf[:, i].sum() - conf[i, i] for i in range(n_cls))
    FN_total = sum(conf[i, :].sum() - conf[i, i] for i in range(n_cls))
    TP_total = np.diag(conf).sum()
    TN_total = conf.sum() - (FP_total + FN_total + TP_total)

    micro_FPR = FP_total / (FP_total + TN_total) if (FP_total + TN_total) > 0 else 0.0
    micro_FNR = FN_total / (FN_total + TP_total) if (FN_total + TP_total) > 0 else 0.0
    macro_FPR = float(np.mean(class_FPR))
    macro_FNR = float(np.mean(class_FNR))
    accuracy  = accuracy_score(y_true, y_pred)
    f1_macro  = f1_score(y_true, y_pred, average='macro')

    print(f"\n{'='*50}")
    print(f"[{model_name}] 평가 결과")
    print(f"{'='*50}")
    print(f"  Accuracy      : {accuracy:.4f}")
    print(f"  F1 (macro)    : {f1_macro:.4f}")
    print(f"  Micro FPR(오탐): {micro_FPR:.4f}")
    print(f"  Micro FNR(미탐): {micro_FNR:.4f}")
    print(f"  Macro FPR     : {macro_FPR:.4f}")
    print(f"  Macro FNR     : {macro_FNR:.4f}")
    print(f"\n  Confusion Matrix:\n{conf}")
    for i in range(n_cls):
        print(f"  Class {i} → FPR: {class_FPR[i]:.4f}, FNR: {class_FNR[i]:.4f}")

    return {
        "model"      : model_name,
        "accuracy"   : accuracy,
        "f1_macro"   : f1_macro,
        "micro_FPR"  : micro_FPR,
        "micro_FNR"  : micro_FNR,
        "macro_FPR"  : macro_FPR,
        "macro_FNR"  : macro_FNR,
        "conf_matrix": conf,
        "class_FPR"  : class_FPR,
        "class_FNR"  : class_FNR,
    }


# ────────────────────────────────────────────────
# BiGRU 모델 정의
# ────────────────────────────────────────────────

class BiGRUClassifier(nn.Module):
    """
    Bidirectional GRU 분류 모델.

    입력 shape: [B, feature_dim] → unsqueeze(1) → [B, 1, feature_dim]
    (feature 전체를 1개의 시퀀스 스텝으로 취급)

    Args:
        input_size  : 입력 피처 차원
        hidden_size : GRU 히든 차원 (default 64)
        num_layers  : GRU 레이어 수 (default 2)
        num_classes : 분류 클래스 수
        dropout     : Dropout 비율 (default 0.2)
    """
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, num_classes: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, feature_dim] → [B, 1, feature_dim]
        x   = x.unsqueeze(1)
        out, _ = self.gru(x)
        out = out[:, -1, :]   # 마지막 시점
        return self.fc(out)


# ────────────────────────────────────────────────
# 모델별 학습/평가 함수
# ────────────────────────────────────────────────

def _run_random_forest(X_train, X_test, y_train, y_test, config) -> dict:
    print("\n[RandomForest] 학습 시작")
    rf = RandomForestClassifier(
        n_estimators = config.get("rf_n_estimators", 100),
        random_state = config.get("random_state",    42),
        n_jobs       = -1,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return _compute_metrics("RandomForest", y_test, y_pred), rf


def _run_xgboost(X_train, X_test, y_train, y_test, config) -> dict:
    print("\n[XGBoost] 학습 시작")
    num_classes = len(np.unique(y_train))
    objective   = "binary:logistic" if num_classes == 2 else "multi:softmax"
    eval_metric = "logloss"         if num_classes == 2 else "mlogloss"

    xgb_clf = xgb.XGBClassifier(
        objective         = objective,
        eval_metric       = eval_metric,
        n_estimators      = config.get("xgb_n_estimators",  200),
        learning_rate     = config.get("xgb_lr",            0.1),
        max_depth         = config.get("xgb_max_depth",     6),
        num_class         = num_classes if num_classes > 2 else None,
        random_state      = config.get("random_state",      42),
        use_label_encoder = False,
    )
    xgb_clf.fit(X_train, y_train)
    y_pred = xgb_clf.predict(X_test)
    return _compute_metrics("XGBoost", y_test, y_pred), xgb_clf


def _run_bigru(X_train, X_test, y_train, y_test, config) -> dict:
    print("\n[BiGRU] 학습 시작")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tensor 변환
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_te = torch.tensor(X_test,  dtype=torch.float32)
    y_te = torch.tensor(y_test,  dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size = config.get("clf_batch_size", 64),
        shuffle    = True,
    )
    test_loader = DataLoader(
        TensorDataset(X_te, y_te),
        batch_size = config.get("clf_batch_size", 64),
        shuffle    = False,
    )

    num_classes = len(np.unique(y_train))
    model = BiGRUClassifier(
        input_size  = X_train.shape[1],
        hidden_size = config.get("gru_hidden",  64),
        num_layers  = config.get("gru_layers",  2),
        num_classes = num_classes,
        dropout     = config.get("gru_dropout", 0.2),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=config.get("gru_lr", 0.001))

    # 학습
    n_epochs = config.get("gru_epochs", 10)
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch [{epoch+1}/{n_epochs}] "
              f"Loss: {total_loss/len(train_loader):.4f}")

    # 평가
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = torch.argmax(model(X_batch.to(device)), dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y_batch.numpy())

    return _compute_metrics("BiGRU", np.array(y_true), np.array(y_pred)), model


# ────────────────────────────────────────────────
# 메인 분류 함수
# ────────────────────────────────────────────────

def classify(config: dict,
             X: np.ndarray,
             y: np.ndarray) -> dict:
    """
    세 분류 모델(RF / XGBoost / BiGRU)을 동일 데이터로 학습·평가.

    Args:
        config:
            - test_size        : train/test 분할 비율 (default 0.3)
            - random_state     : 랜덤 시드 (default 42)
            - max_per_class    : 클래스별 최대 샘플 수 (None이면 전체 사용)
            - rf_n_estimators  : RF 트리 수 (default 100)
            - xgb_n_estimators : XGB 트리 수 (default 200)
            - xgb_lr           : XGB 학습률 (default 0.1)
            - xgb_max_depth    : XGB 트리 깊이 (default 6)
            - gru_hidden       : BiGRU 히든 차원 (default 64)
            - gru_layers       : BiGRU 레이어 수 (default 2)
            - gru_epochs       : BiGRU 학습 epoch (default 10)
            - gru_lr           : BiGRU 학습률 (default 0.001)
            - gru_dropout      : BiGRU dropout (default 0.2)
            - clf_batch_size   : BiGRU 배치 크기 (default 64)
            - models           : 실행할 모델 리스트
                                 (default ['rf', 'xgb', 'bigru'])
        X : [N, feature_dim]  원본 + 증강 합산 피처
        y : [N]               레이블

    Returns:
        {
            'rf'   : {'metrics': dict, 'model': RandomForestClassifier},
            'xgb'  : {'metrics': dict, 'model': XGBClassifier},
            'bigru': {'metrics': dict, 'model': BiGRUClassifier},
        }
    """

    # ── 클래스별 최대 샘플 수 제한 ────────────────
    max_per_class = config.get("max_per_class", None)
    if max_per_class:
        df = pd.DataFrame(X)
        df["__label__"] = y
        df = (df.groupby("__label__", group_keys=False)
                .apply(lambda g: g.sample(
                    n          = min(len(g), max_per_class),
                    random_state = config.get("random_state", 42),
                )))
        df = df.sample(frac=1, random_state=config.get("random_state", 42)
                       ).reset_index(drop=True)
        y = df["__label__"].values
        X = df.drop(columns=["__label__"]).values
        print(f"클래스별 최대 {max_per_class}개로 샘플링 완료")

    print(f"\n전체 샘플: {len(X):,}개 | 클래스 수: {len(np.unique(y))}")
    for cls in np.unique(y):
        print(f"  Label {cls}: {(y==cls).sum():,}개")

    # ── 정규화 [-1, 1] ────────────────────────────
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)

    # ── Train / Test 분할 ─────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = config.get("test_size",    0.3),
        random_state = config.get("random_state", 42),
        stratify     = y,
    )
    print(f"\nTrain: {len(X_train):,}개 | Test: {len(X_test):,}개")

    # ── 모델 실행 ─────────────────────────────────
    models_to_run = config.get("models", ["rf", "xgb", "bigru"])
    results = {}

    if "rf" in models_to_run:
        metrics, model = _run_random_forest(X_train, X_test, y_train, y_test, config)
        results["rf"] = {"metrics": metrics, "model": model}

    if "xgb" in models_to_run:
        metrics, model = _run_xgboost(X_train, X_test, y_train, y_test, config)
        results["xgb"] = {"metrics": metrics, "model": model}

    if "bigru" in models_to_run:
        metrics, model = _run_bigru(X_train, X_test, y_train, y_test, config)
        results["bigru"] = {"metrics": metrics, "model": model}

    # ── 모델 비교 요약 ────────────────────────────
    print(f"\n{'='*50}")
    print("전체 모델 비교")
    print(f"{'='*50}")
    print(f"  {'모델':<12} {'Accuracy':>10} {'F1(macro)':>10} "
          f"{'FPR(micro)':>12} {'FNR(micro)':>12}")
    print(f"  {'-'*58}")
    for key, val in results.items():
        m = val["metrics"]
        print(f"  {m['model']:<12} {m['accuracy']:>10.4f} {m['f1_macro']:>10.4f} "
              f"{m['micro_FPR']:>12.4f} {m['micro_FNR']:>12.4f}")

    return results
