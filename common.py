"""
- 데이터 로드/전처리
- Dataset 클래스
- CNN 모델 (실험 플래그로 Dropout/BatchNorm on-off)
- 평가 함수 (Accuracy/Recall/Macro-F1)
- 실험 결과 로깅 함수

이 파일은 baseline 이후 실험 과정에서 거의 건드리지 않는 걸 목표로 함.
실험별 변경(Early Stopping, Weight Decay, Weighted Loss 등)은 train.py에서 처리.
"""

import os
import pickle
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from skimage.transform import resize
from torch.utils.data import Dataset
from sklearn.metrics import recall_score, f1_score, classification_report

# LSWMD.pkl이 구버전 pandas(pandas.indexes.base 경로)로 저장되어 있어서
# 최신 pandas(pandas.core.indexes.base)에서 바로 못 읽는 문제 우회용 별칭 등록
import pandas.core.indexes.base as _pd_indexes_base
import pandas.core.indexes as _pd_indexes
sys.modules.setdefault('pandas.indexes', _pd_indexes)
sys.modules.setdefault('pandas.indexes.base', _pd_indexes_base)

# ---------------------------------------------------------
# 상수
# ---------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAPPING_TYPE = {
    'Center': 0, 'Donut': 1, 'Edge-Loc': 2, 'Edge-Ring': 3, 'Loc': 4,
    'Near-full': 5, 'Random': 6, 'Scratch': 7, 'none': 8
}
CLASS_NAMES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc',
               'Near-full', 'Random', 'Scratch', 'none']

# Valid-Recall(소수) 계산에 쓰는 하위 3개 클래스 (고정)
MINOR_CLASSES = ['Near-full', 'Donut', 'Random']


# ---------------------------------------------------------
# 데이터 로드 & 전처리
# ---------------------------------------------------------
def _install_pandas_pickle_compat_shims():
    """
    오래된 pandas pickle이 참조하는 구버전 모듈 경로를 현재 pandas 3.x에 매핑.

    LSWMD.pkl 같은 예전 pickle은 pandas 0.x/1.x 시절의 내부 경로를 포함할 수 있어
    pandas 3.x에서 바로 읽으면 ModuleNotFoundError가 날 수 있다.
    """
    if "pandas.indexes" not in sys.modules or "pandas.indexes.base" not in sys.modules:
        import pandas.core.indexes as pandas_indexes
        import pandas.core.indexes.base as pandas_indexes_base
        sys.modules.setdefault("pandas.indexes", pandas_indexes)
        sys.modules.setdefault("pandas.indexes.base", pandas_indexes_base)


def preprocess_wafer_map(wafer_map, target_size=(64, 64)):
    """
    웨이퍼 맵 이미지를 target_size로 리사이징.
    order=0(Nearest Neighbor)로 0/1/2 정수 라벨이 흐트러지지 않게 함.
    """
    resized = resize(wafer_map, target_size,
                      order=0, preserve_range=True, anti_aliasing=False)
    return resized.astype(int)


def load_dataframe(pkl_path="./data/LSWMD.pkl"):
    """
    원본 pkl을 읽어 라벨 전처리까지 마친 DataFrame 반환.
    """
    _install_pandas_pickle_compat_shims()
    with open(pkl_path, "rb") as f:
        try:
            df = pickle.load(f)
        except UnicodeDecodeError:
            # 아주 오래된 pickle은 latin1 인코딩이 필요할 수 있다.
            f.seek(0)
            df = pickle.load(f, encoding="latin1")

    df = df.drop(['waferIndex', 'dieSize', 'lotName'], axis=1)
    df['failureType'] = df['failureType'].apply(lambda x: x[0][0] if len(x) > 0 else 'none')
    df = df[df['failureType'] != '']  # 라벨 없는 데이터 제외

    return df


def build_xy(df, none_sample_n=5000, random_state=42):
    """
    Diet(정상 데이터 샘플링) + 전처리 + X, y 배열 생성.
    """
    df_failure = df[df['failureType'] != 'none'].copy()
    df_none = df[df['failureType'] == 'none'].sample(n=none_sample_n, random_state=random_state).copy()

    df_reduced = pd.concat([df_failure, df_none]).reset_index(drop=True)

    resized_list = df_reduced['waferMap'].apply(lambda x: preprocess_wafer_map(x)).tolist()
    X = np.array(resized_list, dtype=np.float32)
    X = X[:, np.newaxis, :, :]  # (N, 1, H, W)

    y = df_reduced['failureType'].replace(MAPPING_TYPE).values.astype(int)

    return X, y


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
class WaferDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.FloatTensor(images)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# ---------------------------------------------------------
# 모델 (실험 플래그로 옵션 on/off)
# ---------------------------------------------------------
class WaferCNN(nn.Module):
    def __init__(self, use_dropout=False, dropout_p=0.4, use_batchnorm=False):
        super(WaferCNN, self).__init__()

        layer1_modules = [nn.Conv2d(1, 32, kernel_size=3, padding=0)]  # 64 -> 62
        if use_batchnorm:
            layer1_modules.append(nn.BatchNorm2d(32))
        layer1_modules += [nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)]  # 62 -> 31
        self.layer1 = nn.Sequential(*layer1_modules)

        layer2_modules = [nn.Conv2d(32, 64, kernel_size=3, padding=0)]  # 31 -> 29
        if use_batchnorm:
            layer2_modules.append(nn.BatchNorm2d(64))
        layer2_modules += [nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)]  # 29 -> 14
        self.layer2 = nn.Sequential(*layer2_modules)

        fc_modules = [nn.Flatten(), nn.Linear(64 * 14 * 14, 64), nn.ReLU()]
        if use_dropout:
            fc_modules.append(nn.Dropout(dropout_p))
        fc_modules.append(nn.Linear(64, 9))
        self.fc = nn.Sequential(*fc_modules)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc(out)
        return out


# ---------------------------------------------------------
# 평가
# ---------------------------------------------------------
@torch.no_grad()
def evaluate_metrics(model, data_loader, class_names=CLASS_NAMES, verbose=True):
    """
    Accuracy, 클래스별 Recall, Macro-F1을 계산.
    반환: dict (accuracy, macro_f1, recall_per_class: {class_name: recall})
    """
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in data_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    recall_per_class = recall_score(all_labels, all_preds, average=None, labels=range(len(class_names)))
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    recall_dict = {name: r for name, r in zip(class_names, recall_per_class)}

    if verbose:
        for name, r in recall_dict.items():
            print(f"  {name} Recall: {r:.4f}")
        print(f"  Macro-F1: {macro_f1:.4f}")
        print(f"  Accuracy: {accuracy:.2f}%")
        print("\n" + classification_report(all_labels, all_preds, target_names=class_names))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "recall_per_class": recall_dict,
    }


def get_class_weights(y_train, smoothing="none"):
    """
    클래스 가중치 계산. smoothing으로 극단적 가중치를 완화할 수 있음.
    - "none": sklearn의 표준 balanced 가중치 그대로 (exp5에서 사용, 소수 클래스에 과도하게 큰 가중치)
    - "sqrt": 가중치에 제곱근을 취해 완화 (효과는 유지하되 과교정 방지, 널리 쓰이는 표준적인 방법)
    """
    from sklearn.utils.class_weight import compute_class_weight
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

    if smoothing == "sqrt":
        weights = np.sqrt(weights)

    return torch.FloatTensor(weights)


def minor_recall_mean(recall_dict, minor_classes=MINOR_CLASSES):
    """클래스 분포 하위 3개(Near-full, Donut, Random) Recall 평균."""
    return float(np.mean([recall_dict[c] for c in minor_classes]))


# ---------------------------------------------------------
# 결과 로깅
# ---------------------------------------------------------
def log_experiment(exp_name, config, train_metrics, valid_metrics,
                    csv_path="results/experiment_results.csv"):
    """
    실험 결과를 csv에 한 줄씩 append.
    train_metrics, valid_metrics: evaluate_metrics() 반환값
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    row = {
        "experiment": exp_name,
        **config,
        "train_accuracy": train_metrics["accuracy"],
        "valid_accuracy": valid_metrics["accuracy"],
        "valid_macro_f1": valid_metrics["macro_f1"],
        "valid_recall_minor": minor_recall_mean(valid_metrics["recall_per_class"]),
    }
    for cls, r in valid_metrics["recall_per_class"].items():
        row[f"valid_recall_{cls}"] = r

    df_row = pd.DataFrame([row])
    header = not os.path.exists(csv_path)
    df_row.to_csv(csv_path, mode="a", header=header, index=False)
    print(f"[log_experiment] '{exp_name}' 결과가 {csv_path}에 저장됨")