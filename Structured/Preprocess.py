
"""
preprocess.py
-------------
데이터 전처리 파이프라인 모듈.

처리 순서:
  1. CSV 로드
  2. NaN 처리        (평균 / 중앙값 / 최빈값)
  3. 이상치 처리     (IQR / Z-score 감지 → drop / clip)
  4. 레이블 축소     ([[0,1],[2]] 형태 리스트 입력)
  5. Min-Max 정규화  ([-1, 1] 범위)
  6. CustomDataset / DataLoader 생성
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# ────────────────────────────────────────────────
# CustomDataset
# ────────────────────────────────────────────────

class CustomDataset(Dataset):
    """
    원본 코드의 CustomDataset 그대로 유지.
    정규화는 preprocess() 내부에서 완료 후 전달.
    """
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data   = torch.tensor(data,   dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ────────────────────────────────────────────────
# 내부 유틸
# ────────────────────────────────────────────────

def _ask(prompt: str, choices: list) -> str:
    """선택지를 보여주고 유효한 답변이 올 때까지 반복 입력."""
    choice_str = " / ".join(f"[{c}]" for c in choices)
    while True:
        ans = input(f"{prompt} {choice_str}: ").strip().lower()
        if ans in [c.lower() for c in choices]:
            return ans
        print(f"  ⚠ 다시 입력해주세요. 선택지: {choices}")


def _handle_nan(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """NaN 처리."""
    before = df.isnull().sum().sum()
    if before == 0:
        print("  ✅ NaN 없음")
        return df

    if method == "mean":
        df = df.fillna(df.mean())
    elif method == "median":
        df = df.fillna(df.median())
    elif method == "mode":
        df = df.fillna(df.mode().iloc[0])

    print(f"  ✅ NaN {before}개 → {method}으로 대체 완료")
    return df


def _detect_outliers_iqr(df: pd.DataFrame, threshold: float = 1.5) -> pd.Series:
    """IQR 기반 이상치 행 마스크 반환."""
    Q1, Q3 = df.quantile(0.25), df.quantile(0.75)
    IQR    = Q3 - Q1
    mask   = ((df < (Q1 - threshold * IQR)) | (df > (Q3 + threshold * IQR))).any(axis=1)
    return mask


def _detect_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.Series:
    """Z-score 기반 이상치 행 마스크 반환."""
    z    = (df - df.mean()) / (df.std() + 1e-8)
    mask = (z.abs() > threshold).any(axis=1)
    return mask


def _handle_outliers(df: pd.DataFrame, labels: pd.Series,
                     method: str, action: str) -> tuple:
    """
    이상치 감지 후 drop 또는 clip 처리.

    Args:
        method : 'iqr' | 'zscore' | 'both'
        action : 'drop' | 'clip'
    Returns:
        (처리된 df, 처리된 labels)
    """
    if method == "iqr":
        mask = _detect_outliers_iqr(df)
    elif method == "zscore":
        mask = _detect_outliers_zscore(df)
    else:  # both → 둘 다 이상치로 감지된 행만
        mask = _detect_outliers_iqr(df) & _detect_outliers_zscore(df)

    n_outliers = mask.sum()
    print(f"  🔍 이상치 감지 ({method}): {n_outliers}행")

    if n_outliers == 0:
        return df, labels

    if action == "drop":
        df     = df[~mask].reset_index(drop=True)
        labels = labels[~mask].reset_index(drop=True)
        print(f"  ✅ {n_outliers}행 제거 → 남은 샘플: {len(df)}")
    else:  # clip
        Q1, Q3 = df.quantile(0.25), df.quantile(0.75)
        IQR    = Q3 - Q1
        df     = df.clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR, axis=1)
        print(f"  ✅ {n_outliers}행 클리핑 처리 완료")

    return df, labels


def _reduce_labels(labels: pd.Series, groups: list) -> pd.Series:
    """
    레이블 그룹 병합.

    Args:
        groups : [[0,1], [2], [3,4]] 형태
                 리스트 순서가 새 레이블 인덱스 (0, 1, 2, ...)

    Example:
        groups = [[0,1], [2]]
        → 원본 0,1 → 새 레이블 0
          원본 2   → 새 레이블 1
    """
    mapping = {}
    for new_label, group in enumerate(groups):
        for orig_label in group:
            mapping[orig_label] = new_label

    # 매핑에 없는 레이블 확인
    unmapped = set(labels.unique()) - set(mapping.keys())
    if unmapped:
        raise ValueError(f"매핑되지 않은 레이블: {unmapped}\ngroups를 확인해주세요.")

    reduced = labels.map(mapping)
    print(f"  ✅ 레이블 매핑 완료: {mapping}")
    return reduced


def _normalize(df: pd.DataFrame):
    """
    원본 코드와 동일한 Min-Max 정규화 [-1, 1].
      x = 2 * (x - x_min) / (x_max - x_min + 1e-8) - 1

    Returns:
        normalized_df, x_min, x_max
    """
    x_min = df.min()
    x_max = df.max()
    df_norm = 2 * (df - x_min) / (x_max - x_min + 1e-8) - 1
    return df_norm, x_min, x_max


# ────────────────────────────────────────────────
# 메인 전처리 함수
# ────────────────────────────────────────────────

def preprocess(config: dict) -> dict:
    """
    전체 전처리 파이프라인.
    설정값이 없으면 런타임에 사용자에게 직접 질문.

    Args:
        config: pipeline.py 에서 전달하는 딕셔너리.
                아래 키가 없으면 런타임에 질문.
            - data_path    : 피처 CSV 경로
            - label_path   : 레이블 CSV 경로
            - label_col    : 레이블 컬럼명 (default 'label')
            - label_groups : 레이블 축소 그룹 ex) [[0,1],[2]]
                             None 이면 축소 안 함
            - nan_method   : 'mean' | 'median' | 'mode'
            - outlier_method: 'iqr' | 'zscore' | 'both'
            - outlier_action: 'drop' | 'clip' | 'none'
            - batch_size   : DataLoader 배치 크기 (default 64)
            - num_workers  : DataLoader worker 수 (default 4)

    Returns:
        {
            'dataloader' : DataLoader,
            'dataset'    : CustomDataset,
            'x_min'      : pd.Series,   # 역정규화용
            'x_max'      : pd.Series,   # 역정규화용
            'num_classes': int,
            'feature_dim': int,
            'label_groups': list,
        }
    """

    # ── 1. CSV 로드 ────────────────────────────────
    print("\n[전처리 1/5] 데이터 로드")
    x      = pd.read_csv(config["data_path"])
    labels = pd.read_csv(config["label_path"])
    label_col = config.get("label_col", "label")
    labels = labels[label_col]

    print(f"  피처 shape : {x.shape}")
    print(f"  레이블 분포:\n{labels.value_counts().sort_index().to_string()}")

    # ── 2. NaN 처리 ────────────────────────────────
    print("\n[전처리 2/5] NaN 처리")
    nan_method = config.get("nan_method") or _ask(
        "NaN 처리 방법을 선택하세요.", ["mean", "median", "mode"]
    )
    x = _handle_nan(x, nan_method)

    # ── 3. 이상치 처리 ─────────────────────────────
    print("\n[전처리 3/5] 이상치 처리")
    outlier_action = config.get("outlier_action") or _ask(
        "이상치 처리 방법을 선택하세요.", ["drop", "clip", "none"]
    )

    if outlier_action != "none":
        outlier_method = config.get("outlier_method") or _ask(
            "이상치 감지 기준을 선택하세요.", ["iqr", "zscore", "both"]
        )
        x, labels = _handle_outliers(x, labels, outlier_method, outlier_action)
    else:
        print("  ⏭ 이상치 처리 건너뜀")

    # ── 4. 레이블 축소 ─────────────────────────────
    print("\n[전처리 4/5] 레이블 축소")
    label_groups = config.get("label_groups")

    if label_groups is None:
        do_reduce = _ask("레이블 축소를 진행할까요?", ["yes", "no"])
        if do_reduce == "yes":
            print("  병합할 레이블 그룹을 입력하세요.")
            print("  예시) [[0,1],[2]]  →  0,1을 묶어 새 레이블 0, 2를 새 레이블 1로")
            raw = input("  입력: ").strip()
            import ast
            label_groups = ast.literal_eval(raw)

    if label_groups:
        labels = _reduce_labels(labels, label_groups)
    else:
        print("  ⏭ 레이블 축소 건너뜀")

    # 최종 레이블 분포 출력
    print(f"  최종 레이블 분포:\n{labels.value_counts().sort_index().to_string()}")
    num_classes = labels.nunique()

    # ── 5. 정규화 ──────────────────────────────────
    print("\n[전처리 5/5] Min-Max 정규화 [-1, 1]")
    x_norm, x_min, x_max = _normalize(x)
    print(f"  ✅ 정규화 완료 | 범위: [{x_norm.min().min():.2f}, {x_norm.max().max():.2f}]")

    # ── DataLoader 생성 ────────────────────────────
    dataset    = CustomDataset(x_norm.values, labels.values)
    dataloader = DataLoader(
        dataset,
        batch_size  = config.get("batch_size",  64),
        shuffle     = True,
        num_workers = config.get("num_workers",  4),
        pin_memory  = True,
    )

    print(f"\n✅ 전처리 완료")
    print(f"   샘플 수    : {len(dataset)}")
    print(f"   피처 차원  : {x_norm.shape[1]}")
    print(f"   클래스 수  : {num_classes}")
    print(f"   배치 수    : {len(dataloader)}")

    return {
        "dataloader"  : dataloader,
        "dataset"     : dataset,
        "x_min"       : x_min,
        "x_max"       : x_max,
        "num_classes" : num_classes,
        "feature_dim" : x_norm.shape[1],
        "label_groups": label_groups,
    }

"""
preprocess.py
-------------
데이터 전처리 파이프라인 모듈.

처리 순서:
  1. CSV 로드
  2. NaN 처리        (평균 / 중앙값 / 최빈값)
  3. 이상치 처리     (IQR / Z-score 감지 → drop / clip)
  4. 레이블 축소     ([[0,1],[2]] 형태 리스트 입력)
  5. Min-Max 정규화  ([-1, 1] 범위)
  6. CustomDataset / DataLoader 생성
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# ────────────────────────────────────────────────
# CustomDataset
# ────────────────────────────────────────────────

class CustomDataset(Dataset):
    """
    원본 코드의 CustomDataset 그대로 유지.
    정규화는 preprocess() 내부에서 완료 후 전달.
    """
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data   = torch.tensor(data,   dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ────────────────────────────────────────────────
# 내부 유틸
# ────────────────────────────────────────────────

def _ask(prompt: str, choices: list) -> str:
    """선택지를 보여주고 유효한 답변이 올 때까지 반복 입력."""
    choice_str = " / ".join(f"[{c}]" for c in choices)
    while True:
        ans = input(f"{prompt} {choice_str}: ").strip().lower()
        if ans in [c.lower() for c in choices]:
            return ans
        print(f"  ⚠ 다시 입력해주세요. 선택지: {choices}")


def _handle_nan(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """NaN 처리."""
    before = df.isnull().sum().sum()
    if before == 0:
        print("  ✅ NaN 없음")
        return df

    if method == "mean":
        df = df.fillna(df.mean())
    elif method == "median":
        df = df.fillna(df.median())
    elif method == "mode":
        df = df.fillna(df.mode().iloc[0])

    print(f"  ✅ NaN {before}개 → {method}으로 대체 완료")
    return df


def _detect_outliers_iqr(df: pd.DataFrame, threshold: float = 1.5) -> pd.Series:
    """IQR 기반 이상치 행 마스크 반환."""
    Q1, Q3 = df.quantile(0.25), df.quantile(0.75)
    IQR    = Q3 - Q1
    mask   = ((df < (Q1 - threshold * IQR)) | (df > (Q3 + threshold * IQR))).any(axis=1)
    return mask


def _detect_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.Series:
    """Z-score 기반 이상치 행 마스크 반환."""
    z    = (df - df.mean()) / (df.std() + 1e-8)
    mask = (z.abs() > threshold).any(axis=1)
    return mask


def _handle_outliers(df: pd.DataFrame, labels: pd.Series,
                     method: str, action: str) -> tuple:
    """
    이상치 감지 후 drop 또는 clip 처리.

    Args:
        method : 'iqr' | 'zscore' | 'both'
        action : 'drop' | 'clip'
    Returns:
        (처리된 df, 처리된 labels)
    """
    if method == "iqr":
        mask = _detect_outliers_iqr(df)
    elif method == "zscore":
        mask = _detect_outliers_zscore(df)
    else:  # both → 둘 다 이상치로 감지된 행만
        mask = _detect_outliers_iqr(df) & _detect_outliers_zscore(df)

    n_outliers = mask.sum()
    print(f"  🔍 이상치 감지 ({method}): {n_outliers}행")

    if n_outliers == 0:
        return df, labels

    if action == "drop":
        df     = df[~mask].reset_index(drop=True)
        labels = labels[~mask].reset_index(drop=True)
        print(f"  ✅ {n_outliers}행 제거 → 남은 샘플: {len(df)}")
    else:  # clip
        Q1, Q3 = df.quantile(0.25), df.quantile(0.75)
        IQR    = Q3 - Q1
        df     = df.clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR, axis=1)
        print(f"  ✅ {n_outliers}행 클리핑 처리 완료")

    return df, labels


def _reduce_labels(labels: pd.Series, groups: list) -> pd.Series:
    """
    레이블 그룹 병합.

    Args:
        groups : [[0,1], [2], [3,4]] 형태
                 리스트 순서가 새 레이블 인덱스 (0, 1, 2, ...)

    Example:
        groups = [[0,1], [2]]
        → 원본 0,1 → 새 레이블 0
          원본 2   → 새 레이블 1
    """
    mapping = {}
    for new_label, group in enumerate(groups):
        for orig_label in group:
            mapping[orig_label] = new_label

    # 매핑에 없는 레이블 확인
    unmapped = set(labels.unique()) - set(mapping.keys())
    if unmapped:
        raise ValueError(f"매핑되지 않은 레이블: {unmapped}\ngroups를 확인해주세요.")

    reduced = labels.map(mapping)
    print(f"  ✅ 레이블 매핑 완료: {mapping}")
    return reduced


def _normalize(df: pd.DataFrame):
    """
    원본 코드와 동일한 Min-Max 정규화 [-1, 1].
      x = 2 * (x - x_min) / (x_max - x_min + 1e-8) - 1

    Returns:
        normalized_df, x_min, x_max
    """
    x_min = df.min()
    x_max = df.max()
    df_norm = 2 * (df - x_min) / (x_max - x_min + 1e-8) - 1
    return df_norm, x_min, x_max


# ────────────────────────────────────────────────
# 메인 전처리 함수
# ────────────────────────────────────────────────

def preprocess(config: dict) -> dict:
    """
    전체 전처리 파이프라인.
    설정값이 없으면 런타임에 사용자에게 직접 질문.

    Args:
        config: pipeline.py 에서 전달하는 딕셔너리.
                아래 키가 없으면 런타임에 질문.
            - data_path    : 피처 CSV 경로
            - label_path   : 레이블 CSV 경로
            - label_col    : 레이블 컬럼명 (default 'label')
            - label_groups : 레이블 축소 그룹 ex) [[0,1],[2]]
                             None 이면 축소 안 함
            - nan_method   : 'mean' | 'median' | 'mode'
            - outlier_method: 'iqr' | 'zscore' | 'both'
            - outlier_action: 'drop' | 'clip' | 'none'
            - batch_size   : DataLoader 배치 크기 (default 64)
            - num_workers  : DataLoader worker 수 (default 4)

    Returns:
        {
            'dataloader' : DataLoader,
            'dataset'    : CustomDataset,
            'x_min'      : pd.Series,   # 역정규화용
            'x_max'      : pd.Series,   # 역정규화용
            'num_classes': int,
            'feature_dim': int,
            'label_groups': list,
        }
    """

    # ── 1. CSV 로드 ────────────────────────────────
    print("\n[전처리 1/5] 데이터 로드")
    x      = pd.read_csv(config["data_path"])
    labels = pd.read_csv(config["label_path"])
    label_col = config.get("label_col", "label")
    labels = labels[label_col]

    print(f"  피처 shape : {x.shape}")
    print(f"  레이블 분포:\n{labels.value_counts().sort_index().to_string()}")

    # ── 2. NaN 처리 ────────────────────────────────
    print("\n[전처리 2/5] NaN 처리")
    nan_method = config.get("nan_method") or _ask(
        "NaN 처리 방법을 선택하세요.", ["mean", "median", "mode"]
    )
    x = _handle_nan(x, nan_method)

    # ── 3. 이상치 처리 ─────────────────────────────
    print("\n[전처리 3/5] 이상치 처리")
    outlier_action = config.get("outlier_action") or _ask(
        "이상치 처리 방법을 선택하세요.", ["drop", "clip", "none"]
    )

    if outlier_action != "none":
        outlier_method = config.get("outlier_method") or _ask(
            "이상치 감지 기준을 선택하세요.", ["iqr", "zscore", "both"]
        )
        x, labels = _handle_outliers(x, labels, outlier_method, outlier_action)
    else:
        print("  ⏭ 이상치 처리 건너뜀")

    # ── 4. 레이블 축소 ─────────────────────────────
    print("\n[전처리 4/5] 레이블 축소")
    label_groups = config.get("label_groups")

    if label_groups is None:
        do_reduce = _ask("레이블 축소를 진행할까요?", ["yes", "no"])
        if do_reduce == "yes":
            print("  병합할 레이블 그룹을 입력하세요.")
            print("  예시) [[0,1],[2]]  →  0,1을 묶어 새 레이블 0, 2를 새 레이블 1로")
            raw = input("  입력: ").strip()
            import ast
            label_groups = ast.literal_eval(raw)

    if label_groups:
        labels = _reduce_labels(labels, label_groups)
    else:
        print("  ⏭ 레이블 축소 건너뜀")

    # 최종 레이블 분포 출력
    print(f"  최종 레이블 분포:\n{labels.value_counts().sort_index().to_string()}")
    num_classes = labels.nunique()

    # ── 5. 정규화 ──────────────────────────────────
    print("\n[전처리 5/5] Min-Max 정규화 [-1, 1]")
    x_norm, x_min, x_max = _normalize(x)
    print(f"  ✅ 정규화 완료 | 범위: [{x_norm.min().min():.2f}, {x_norm.max().max():.2f}]")

    # ── DataLoader 생성 ────────────────────────────
    dataset    = CustomDataset(x_norm.values, labels.values)
    dataloader = DataLoader(
        dataset,
        batch_size  = config.get("batch_size",  64),
        shuffle     = True,
        num_workers = config.get("num_workers",  4),
        pin_memory  = True,
    )

    print(f"\n✅ 전처리 완료")
    print(f"   샘플 수    : {len(dataset)}")
    print(f"   피처 차원  : {x_norm.shape[1]}")
    print(f"   클래스 수  : {num_classes}")
    print(f"   배치 수    : {len(dataloader)}")

    return {
        "dataloader"  : dataloader,
        "dataset"     : dataset,
        "x_min"       : x_min,
        "x_max"       : x_max,
        "num_classes" : num_classes,
        "feature_dim" : x_norm.shape[1],
        "label_groups": label_groups,
    }

