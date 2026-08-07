"""
실험 실행 스크립트

진행 방식:
- 아래 EXP_NAME과 플래그들을 실험마다 하나씩 바꾸고 커밋
- exp1: baseline (3분할만 적용, 모든 옵션 False)
- exp2: USE_EARLY_STOPPING = True
- exp3: USE_DROPOUT = True
- exp4: USE_WEIGHT_DECAY = True
- exp5: USE_WEIGHTED_LOSS = True
- exp6: USE_BATCHNORM = True
- exp7: 전부 True (최종 조합)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from common import (
    DEVICE, CLASS_NAMES,
    load_dataframe, build_xy, WaferDataset, WaferCNN,
    evaluate_metrics, log_experiment,
)

# ===========================================================
# 실험 설정
# ===========================================================
EXP_NAME = "exp2_early_stopping"

USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 5

USE_DROPOUT = False
DROPOUT_P = 0.4

USE_WEIGHT_DECAY = False
WEIGHT_DECAY = 1e-4

USE_WEIGHTED_LOSS = False

USE_BATCHNORM = False

EPOCHS = 10
BATCH_SIZE = 64
LR = 0.001
RANDOM_STATE = 42

# ===========================================================
# 데이터 준비 (Train 56% / Valid 24% / Test 20%)
# ===========================================================
print("데이터 로드 중...")
df = load_dataframe("./data/LSWMD.pkl")
X, y = build_xy(df)
print(f"X shape: {X.shape}, y shape: {y.shape}")

# 1차: (train+valid) 80% vs test 20%
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
# 2차: (train+valid) 80% -> train 70% / valid 30%  => 전체 기준 train 56% / valid 24%
X_train, X_valid, y_train, y_valid = train_test_split(
    X_trainval, y_trainval, test_size=0.3, random_state=RANDOM_STATE, stratify=y_trainval
)

print(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

train_loader = DataLoader(WaferDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(WaferDataset(X_valid, y_valid), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(WaferDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# ===========================================================
# 모델 / 손실함수 / 옵티마이저
# ===========================================================
model = WaferCNN(use_dropout=USE_DROPOUT, dropout_p=DROPOUT_P,
                  use_batchnorm=USE_BATCHNORM).to(DEVICE)
print(model)

if USE_WEIGHTED_LOSS:
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
else:
    criterion = nn.CrossEntropyLoss()

optimizer_kwargs = {"lr": LR}
if USE_WEIGHT_DECAY:
    optimizer_kwargs["weight_decay"] = WEIGHT_DECAY
optimizer = optim.Adam(model.parameters(), **optimizer_kwargs)


# ===========================================================
# 학습 루프 (Early Stopping 옵션 포함)
# ===========================================================
def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs):
    best_valid_f1 = -1
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")

        if USE_EARLY_STOPPING:
            valid_metrics = evaluate_metrics(model, valid_loader, verbose=False)
            valid_f1 = valid_metrics["macro_f1"]
            print(f"           Valid Macro-F1: {valid_f1:.4f}")

            if valid_f1 > best_valid_f1:
                best_valid_f1 = valid_f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping (patience={EARLY_STOPPING_PATIENCE}) at epoch {epoch+1}")
                    break

    if USE_EARLY_STOPPING and best_state is not None:
        model.load_state_dict(best_state)

    return model


print("\n학습 시작...")
model = train_model(model, train_loader, valid_loader, criterion, optimizer, EPOCHS)

# ===========================================================
# 평가 & 결과 저장
# ===========================================================
print("\n[Train 평가]")
train_metrics = evaluate_metrics(model, train_loader, CLASS_NAMES, verbose=False)
print(f"Train Accuracy: {train_metrics['accuracy']:.2f}%")

print("\n[Valid 평가]")
valid_metrics = evaluate_metrics(model, valid_loader, CLASS_NAMES, verbose=True)

config = {
    "use_early_stopping": USE_EARLY_STOPPING,
    "use_dropout": USE_DROPOUT,
    "use_weight_decay": USE_WEIGHT_DECAY,
    "use_weighted_loss": USE_WEIGHTED_LOSS,
    "use_batchnorm": USE_BATCHNORM,
    "epochs_run": EPOCHS,
}
log_experiment(EXP_NAME, config, train_metrics, valid_metrics)

print("\n[Test 평가 - 최종 모델 후보에서만 참고용으로 확인]")
test_metrics = evaluate_metrics(model, test_loader, CLASS_NAMES, verbose=True)
