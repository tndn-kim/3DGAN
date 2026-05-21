import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


'''RF 분류 모델'''
# 데이터 로드
data = pd.read_csv('C:/Users/milab_8/Desktop/day2_new.csv').astype(np.float32)
label = pd.read_csv('C:/Users/milab_8/Desktop/label2_3way.csv')

# 데이터 + 라벨 결합
data['label_num'] = label['label_num']

# 클래스별 샘플링
class_0 = data[data['label_num'] == 0]
class_1 = data[data['label_num'] == 1].sample(n=100000, random_state=42)
class_2 = data[data['label_num'] == 2].sample(n=100000, random_state=42)

# 병합 후 섞기
sampled = pd.concat([class_0, class_1, class_2], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# 라벨 분리
labels = sampled['label_num'].values
sampled = sampled.drop(columns=['label_num'])

# [-1, 1] 정규화
sampled = 2 * (sampled - sampled.min()) / (sampled.max() - sampled.min() + 1e-8) - 1

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(sampled, labels, test_size=0.3, random_state=42, stratify=labels)

# 모델 학습
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 예측
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# 혼동 행렬
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# 클래스별 FPR/FNR 계산
num_classes = conf_matrix.shape[0]
class_FPR, class_FNR = [], []

for i in range(num_classes):
    TP = conf_matrix[i, i]
    FN = np.sum(conf_matrix[i, :]) - TP
    FP = np.sum(conf_matrix[:, i]) - TP
    TN = np.sum(conf_matrix) - (TP + FP + FN)
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    class_FPR.append(FPR)
    class_FNR.append(FNR)
    print(f"Class {i} - FPR: {FPR:.4f}, FNR: {FNR:.4f}")

# 마이크로 평균
FP_total = np.sum([np.sum(conf_matrix[:, i]) - conf_matrix[i, i] for i in range(num_classes)])
FN_total = np.sum([np.sum(conf_matrix[i, :]) - conf_matrix[i, i] for i in range(num_classes)])
TP_total = np.sum(np.diag(conf_matrix))
TN_total = np.sum(conf_matrix) - (FP_total + FN_total + TP_total)
micro_FPR = FP_total / (FP_total + TN_total)
micro_FNR = FN_total / (FN_total + TP_total)
print(f"Micro-averaged FPR: {micro_FPR:.4f}")
print(f"Micro-averaged FNR: {micro_FNR:.4f}")

# 매크로 평균
macro_FPR = np.mean(class_FPR)
macro_FNR = np.mean(class_FNR)
print(f"Macro-averaged FPR: {macro_FPR:.4f}")
print(f"Macro-averaged FNR: {macro_FNR:.4f}")

# F1-score
f1_macro = f1_score(y_test, y_pred, average='macro')
print(f"Macro F1-score: {f1_macro:.4f}")


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# ===============================
# 1.BiGRU 분류모델
# ===============================
data = pd.read_csv('C:/Users/milab_8/Desktop/day2_new.csv').astype(np.float32)
label = pd.read_csv('C:/Users/milab_8/Desktop/label2_3way.csv')

# 데이터 + 라벨 결합
data['label_num'] = label['label_num']

# 클래스별 샘플링
class_0 = data[data['label_num'] == 0]
class_1 = data[data['label_num'] == 1].sample(n=100000, random_state=42)
class_2 = data[data['label_num'] == 2].sample(n=100000, random_state=42)

# 병합 후 섞기
sampled = pd.concat([class_0, class_1, class_2], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# 라벨 분리
labels = sampled['label_num'].values
sampled = sampled.drop(columns=['label_num'])

# [-1, 1] 정규화
sampled = 2 * (sampled - sampled.min()) / (sampled.max() - sampled.min() + 1e-8) - 1 

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(sampled, labels, test_size=0.3, random_state=42, stratify=labels)


# Tensor 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# DataLoader 생성
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                           batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor),
                          batch_size=batch_size, shuffle=False)

# ===============================
# 3. BiGRU 모델 정의
# ===============================
class BiGRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(BiGRUClassifier, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 양방향이므로 *2

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]   # 마지막 시점
        out = self.fc(out)
        return out

# ===============================
# 4. 학습 준비
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiGRUClassifier(input_size=feature_size, hidden_size=64, num_layers=2, num_classes=len(np.unique(y))).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===============================
# 5. 학습 루프
# ===============================
for epoch in range(10):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/10], Loss: {total_loss/len(train_loader):.4f}")
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# ===============================
# 6. 평가
# ===============================
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y_batch.numpy())

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

num_classes = conf_matrix.shape[0]

# 클래스별 FPR/FNR 계산
class_FPR = []
class_FNR = []

for i in range(num_classes):
    TP = conf_matrix[i, i]
    FN = np.sum(conf_matrix[i, :]) - TP
    FP = np.sum(conf_matrix[:, i]) - TP
    TN = np.sum(conf_matrix) - (TP + FP + FN)
    
    # 분모가 0인 경우 예외 처리
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    
    class_FPR.append(FPR)
    class_FNR.append(FNR)
    
    print(f"Class {i} - FPR: {FPR:.4f}, FNR: {FNR:.4f}")

# 마이크로 평균 FPR/FNR (전체 기준)
FP_total = np.sum([np.sum(conf_matrix[:, i]) - conf_matrix[i, i] for i in range(num_classes)])
FN_total = np.sum([np.sum(conf_matrix[i, :]) - conf_matrix[i, i] for i in range(num_classes)])
TP_total = np.sum(np.diag(conf_matrix))
TN_total = np.sum(conf_matrix) - (FP_total + FN_total + TP_total)

micro_FPR = FP_total / (FP_total + TN_total) if (FP_total + TN_total) > 0 else 0.0
micro_FNR = FN_total / (FN_total + TP_total) if (FN_total + TP_total) > 0 else 0.0

print(f"Micro-averaged FPR: {micro_FPR:.4f}")#오탐
print(f"Micro-averaged FNR: {micro_FNR:.4f}")#미탐

# 매크로 평균 FPR/FNR (클래스별 평균)
macro_FPR = np.mean(class_FPR)
macro_FNR = np.mean(class_FNR)

print(f"Macro-averaged FPR: {macro_FPR:.4f}")
print(f"Macro-averaged FNR: {macro_FNR:.4f}")

f1_macro = f1_score(y_test, y_pred, average='macro')
print(f1_macro)

import xgboost as xgb

'''XGBoost 분류 모델'''
data = np.load('C:/Users/milab_8/Desktop/3dGAN_output.npy').astype(np.float32)
label = pd.read_csv('C:/Users/milab_8/Desktop/3dGAN_label.csv')

# 데이터 프레임 생성
sampled = pd.DataFrame(data)
sampled['label_num'] = label['label'].values

# 라벨 분리
labels = sampled['label_num'].values
sampled = sampled.drop(columns=['label_num'])

# 클래스 개수 확인
num_classes = len(np.unique(labels))
print(f"클래스 개수: {num_classes}")
print(f"데이터 형태: {sampled.shape}")
print(f"라벨 분포: {np.bincount(labels.astype(int))}")

# [-1, 1] 정규화 (컬럼별로)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
sampled = scaler.fit_transform(sampled)
sampled = pd.DataFrame(sampled)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(sampled, labels, test_size=0.3, random_state=42, stratify=labels)

# 클래스 개수에 따라 objective 설정
if num_classes == 2:
    objective = 'binary:logistic'
    eval_metric = 'logloss'
else:
    objective = 'multi:softmax'
    eval_metric = 'mlogloss'

xgb_clf = xgb.XGBClassifier(
    objective=objective,
    eval_metric=eval_metric,
    use_label_encoder=False,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    num_class=num_classes if num_classes > 2 else None,
    random_state=42
)


xgb_clf.fit(X_train, y_train)
y_pred = xgb_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# 혼동 행렬
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)


num_classes = conf_matrix.shape[0]

# 클래스별 FPR/FNR 계산
class_FPR = []
class_FNR = []

for i in range(num_classes):
    TP = conf_matrix[i, i]
    FN = np.sum(conf_matrix[i, :]) - TP
    FP = np.sum(conf_matrix[:, i]) - TP
    TN = np.sum(conf_matrix) - (TP + FP + FN)
    
    # 분모가 0인 경우 예외 처리
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    
    class_FPR.append(FPR)
    class_FNR.append(FNR)
    
    print(f"Class {i} - FPR: {FPR:.4f}, FNR: {FNR:.4f}")

# 마이크로 평균 FPR/FNR (전체 기준)
FP_total = np.sum([np.sum(conf_matrix[:, i]) - conf_matrix[i, i] for i in range(num_classes)])
FN_total = np.sum([np.sum(conf_matrix[i, :]) - conf_matrix[i, i] for i in range(num_classes)])
TP_total = np.sum(np.diag(conf_matrix))
TN_total = np.sum(conf_matrix) - (FP_total + FN_total + TP_total)

micro_FPR = FP_total / (FP_total + TN_total) if (FP_total + TN_total) > 0 else 0.0
micro_FNR = FN_total / (FN_total + TP_total) if (FN_total + TP_total) > 0 else 0.0

print(f"Micro-averaged FPR: {micro_FPR:.4f}")#오탐
print(f"Micro-averaged FNR: {micro_FNR:.4f}")#미탐


# 매크로 평균 FPR/FNR (클래스별 평균)
macro_FPR = np.mean(class_FPR)
macro_FNR = np.mean(class_FNR)

print(f"Macro-averaged FPR: {macro_FPR:.4f}")
print(f"Macro-averaged FNR: {macro_FNR:.4f}")

f1_macro = f1_score(y_test, y_pred, average='macro')
print(f"F1-Score (Macro): {f1_macro:.4f}")
print(f1_macro)