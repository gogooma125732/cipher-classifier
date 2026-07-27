from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. 기본 설정
# ============================================================

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# 문제에서 주어진 영어 알파벳 기대 빈도
ENGLISH_FREQ = np.array(
    [
        8.17, 1.50, 2.78, 4.25, 12.70, 2.23, 2.02,
        6.09, 6.97, 0.15, 0.77, 4.03, 2.41,
        6.75, 7.51, 1.93, 0.10, 5.99, 6.33,
        9.06, 2.76, 0.98, 2.36, 0.15, 1.97, 0.07,
    ],
    dtype=float,
)

ENGLISH_FREQ /= ENGLISH_FREQ.sum()

LABEL_NAMES = {
    0: "Caesar",
    1: "Vigenere",
}


# 문제 5의 분류 대상 암호문
UNKNOWN_CIPHERTEXTS = {
    "암호문 1": (
        "NKRRUZNOYOYGIRGYYOIGRIOVNKXGTGREYOYVXUHRKSLUXZNKIXEVZGTGREYOYIUSVKZOZOUTIUTMXGZARG"
        "ZOUTYUTMKZZOTMZNKIUXXKIZGTYCKX"
    ),
    "암호문 2": (
        "ROVVYDRSCSCKMVKCCSMKVMSZROBKXKVICSCZBYLVOWPYBDROMBIZDKXKVICSCMYWZODSDSYXMYXQBKD"
        "EVKDSYXCYXQODDSXQDROMYBBOMDKXCGOB"
    ),
    "암호문 3": (
        "DRKXUIYEPYBIYEBZKBDSMSZKDSYXGOGSCRIYEKVVDROLOCDSXIYEBPEDEBOOXNOKFYBC"
    ),
    "암호문 4": (
        "ZNGTQEUALUXEUAXVGXZOIOVGZOUTCKCOYNEUAGRRZNKHKYZOTEUAXLAZAXKKTJKGBUXY"
    ),
}


# ============================================================
# 2. 문자열 전처리
# ============================================================

def clean_text(text: str) -> str:
    """
    문자열에서 알파벳만 남기고 대문자로 변환한다.
    """
    return "".join(re.findall(r"[A-Z]", text.upper()))


def load_ciphertext(path: str | Path) -> str:
    """
    텍스트 파일을 읽고 암호문을 정제한다.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    text = path.read_text(encoding="utf-8")
    cleaned = clean_text(text)

    if not cleaned:
        raise ValueError(f"파일에 유효한 알파벳 암호문이 없습니다: {path}")

    return cleaned


# ============================================================
# 3. 통계 feature
# ============================================================

def letter_counts(text: str) -> np.ndarray:
    """
    A~Z 각각의 출현 횟수를 반환한다.
    """
    counts = np.zeros(26, dtype=float)

    for char in text:
        counts[ord(char) - ord("A")] += 1

    return counts


def index_of_coincidence(text: str) -> float:
    """
    IC = sum(n_i(n_i-1)) / (N(N-1))
    """
    n = len(text)

    if n < 2:
        return 0.0

    counts = letter_counts(text)
    numerator = np.sum(counts * (counts - 1))

    return float(numerator / (n * (n - 1)))


def shannon_entropy(text: str) -> float:
    """
    알파벳 확률 분포의 Shannon entropy를 계산한다.
    """
    if not text:
        return 0.0

    counts = letter_counts(text)
    probabilities = counts[counts > 0] / len(text)

    return float(-np.sum(probabilities * np.log2(probabilities)))


def caesar_decrypt(text: str, key: int) -> str:
    """
    Caesar 암호를 key만큼 역방향 이동하여 복호화한다.
    """
    result = []

    for char in text:
        value = (ord(char) - ord("A") - key) % 26
        result.append(chr(value + ord("A")))

    return "".join(result)


def chi_squared_english(text: str) -> float:
    """
    문자열의 알파벳 빈도와 영어 기대 빈도의 카이제곱 값을 계산한다.
    """
    n = len(text)

    if n == 0:
        return float("inf")

    observed = letter_counts(text)
    expected = ENGLISH_FREQ * n

    # 기대 빈도가 매우 작은 문자가 있으므로 0 나눗셈 방지
    expected = np.maximum(expected, 1e-12)

    chi_squared = np.sum((observed - expected) ** 2 / expected)

    return float(chi_squared)


def caesar_chi_squared_features(text: str) -> tuple[float, float, float]:
    """
    26개의 Caesar 이동값에 대한 카이제곱 값을 계산한다.

    반환값:
        최소 카이제곱,
        두 번째 최소 카이제곱,
        두 값의 차이
    """
    scores = []

    for key in range(26):
        decrypted = caesar_decrypt(text, key)
        score = chi_squared_english(decrypted)
        scores.append(score)

    scores = sorted(scores)

    minimum = scores[0]
    second_minimum = scores[1]
    gap = second_minimum - minimum

    return minimum, second_minimum, gap


def average_column_ic(text: str, key_length: int) -> float:
    """
    문자열을 key_length개의 부분열로 나눈 뒤
    각 부분열의 IC 평균을 계산한다.
    """
    if key_length <= 0:
        raise ValueError("key_length는 1 이상이어야 합니다.")

    columns = [
        text[offset::key_length]
        for offset in range(key_length)
    ]

    valid_ics = [
        index_of_coincidence(column)
        for column in columns
        if len(column) >= 2
    ]

    if not valid_ics:
        return 0.0

    return float(np.mean(valid_ics))


def lag_coincidence(text: str, lag: int) -> float:
    """
    text[i] == text[i + lag]인 비율을 계산한다.
    반복 키 암호의 주기성을 나타내는 feature로 사용한다.
    """
    if lag <= 0 or len(text) <= lag:
        return 0.0

    matches = sum(
        text[i] == text[i + lag]
        for i in range(len(text) - lag)
    )

    return matches / (len(text) - lag)


def repeated_ngram_ratio(text: str, ngram_size: int = 3) -> float:
    """
    n-gram 중 두 번 이상 등장하는 패턴의 비율을 계산한다.
    """
    total = len(text) - ngram_size + 1

    if total <= 0:
        return 0.0

    ngrams = [
        text[i:i + ngram_size]
        for i in range(total)
    ]

    counts = Counter(ngrams)

    repeated_occurrences = sum(
        count
        for count in counts.values()
        if count >= 2
    )

    return repeated_occurrences / total


def extract_features(text: str) -> np.ndarray:
    """
    하나의 암호문 구간에서 분류 feature를 추출한다.
    """
    text = clean_text(text)

    if len(text) < 2:
        raise ValueError("feature 추출을 위해 최소 2글자가 필요합니다.")

    counts = letter_counts(text)
    frequencies = counts / len(text)
    sorted_frequencies = np.sort(frequencies)[::-1]

    whole_ic = index_of_coincidence(text)
    entropy = shannon_entropy(text)

    max_frequency = sorted_frequencies[0]
    top3_frequency = np.sum(sorted_frequencies[:3])
    frequency_variance = np.var(frequencies)

    minimum_chi, second_chi, chi_gap = caesar_chi_squared_features(text)

    # 길이에 따른 카이제곱 규모 차이를 줄이기 위해 문자 수로 정규화
    minimum_chi_normalized = minimum_chi / len(text)
    second_chi_normalized = second_chi / len(text)
    chi_gap_normalized = chi_gap / len(text)

    column_ics = [
        average_column_ic(text, key_length)
        for key_length in range(2, 6)
    ]

    # 전체 IC에 비해 부분열 IC가 얼마나 증가하는가
    column_ic_gains = [
        column_ic - whole_ic
        for column_ic in column_ics
    ]

    lag_features = [
        lag_coincidence(text, lag)
        for lag in range(1, 6)
    ]

    repeat_ratio_3 = repeated_ngram_ratio(text, 3)
    repeat_ratio_4 = repeated_ngram_ratio(text, 4)

    features = np.concatenate(
        [
            # 알파벳 26개 빈도
            frequencies,

            # 전역 통계
            np.array(
                [
                    whole_ic,
                    entropy,
                    max_frequency,
                    top3_frequency,
                    frequency_variance,
                    minimum_chi_normalized,
                    second_chi_normalized,
                    chi_gap_normalized,
                ]
            ),

            # 키 길이 후보별 부분열 IC
            np.array(column_ics),

            # 전체 IC 대비 부분열 IC 상승량
            np.array(column_ic_gains),

            # lag 1~5 일치 비율
            np.array(lag_features),

            # 반복 n-gram 비율
            np.array(
                [
                    repeat_ratio_3,
                    repeat_ratio_4,
                    len(text),
                ]
            ),
        ]
    )

    return features.astype(float)


# ============================================================
# 4. Sliding window 데이터 생성
# ============================================================

def make_windows(
    text: str,
    window_size: int = 120,
    stride: int = 30,
) -> list[str]:
    """
    긴 암호문을 일정 크기의 겹치는 구간으로 분할한다.
    """
    if window_size <= 0:
        raise ValueError("window_size는 1 이상이어야 합니다.")

    if stride <= 0:
        raise ValueError("stride는 1 이상이어야 합니다.")

    if len(text) < window_size:
        return [text]

    return [
        text[start:start + window_size]
        for start in range(0, len(text) - window_size + 1, stride)
    ]


def split_source_text(
    text: str,
    train_ratio: float = 0.75,
    exclusion_gap: int = 200,
) -> tuple[str, str]:
    """
    동일하거나 겹치는 문자열이 학습·검증 데이터에 동시에 들어가는 것을
    줄이기 위해 원문 자체를 앞부분과 뒷부분으로 분리한다.

    분할 경계 주변 exclusion_gap 글자는 사용하지 않는다.
    """
    split_index = int(len(text) * train_ratio)

    train_end = max(0, split_index - exclusion_gap)
    test_start = min(len(text), split_index + exclusion_gap)

    train_text = text[:train_end]
    test_text = text[test_start:]

    return train_text, test_text


def build_dataset(
    caesar_text: str,
    vigenere_text: str,
    window_size: int = 120,
    stride: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Caesar와 Vigenère 파일로부터 학습·검증 데이터셋을 생성한다.
    """
    caesar_train, caesar_test = split_source_text(caesar_text)
    vigenere_train, vigenere_test = split_source_text(vigenere_text)

    train_windows = []
    train_labels = []
    test_windows = []
    test_labels = []

    for window in make_windows(caesar_train, window_size, stride):
        train_windows.append(window)
        train_labels.append(0)

    for window in make_windows(vigenere_train, window_size, stride):
        train_windows.append(window)
        train_labels.append(1)

    for window in make_windows(caesar_test, window_size, stride):
        test_windows.append(window)
        test_labels.append(0)

    for window in make_windows(vigenere_test, window_size, stride):
        test_windows.append(window)
        test_labels.append(1)

    x_train = np.vstack([
        extract_features(window)
        for window in train_windows
    ])

    y_train = np.array(train_labels, dtype=int)

    x_test = np.vstack([
        extract_features(window)
        for window in test_windows
    ])

    y_test = np.array(test_labels, dtype=int)

    return x_train, y_train, x_test, y_test


# ============================================================
# 5. 짧은 암호문의 예측 안정화
# ============================================================

def prediction_segments(
    text: str,
    minimum_segment_size: int = 60,
) -> list[str]:
    """
    문제 5의 암호문은 학습 window보다 짧을 수 있다.

    전체 문자열뿐 아니라 앞부분, 뒷부분, 홀수/짝수 위치 등의
    여러 view를 생성해 예측 확률을 평균한다.
    """
    text = clean_text(text)
    segments = [text]

    if len(text) >= minimum_segment_size * 2:
        middle = len(text) // 2
        segments.append(text[:middle])
        segments.append(text[middle:])

    # 길이가 충분한 경우 서로 다른 위치의 부분 문자열 추가
    if len(text) >= 80:
        segment_length = max(60, int(len(text) * 0.75))

        segments.append(text[:segment_length])
        segments.append(text[-segment_length:])

    # 위치별 부분열도 주기적 암호 특성을 일부 반영할 수 있음
    even_sequence = text[0::2]
    odd_sequence = text[1::2]

    if len(even_sequence) >= minimum_segment_size:
        segments.append(even_sequence)

    if len(odd_sequence) >= minimum_segment_size:
        segments.append(odd_sequence)

    # 중복 제거
    unique_segments = list(dict.fromkeys(segments))

    return unique_segments


def predict_ciphertext(
    model: Pipeline,
    text: str,
) -> tuple[int, np.ndarray]:
    """
    여러 segment의 예측 확률을 평균하여 최종 클래스를 결정한다.
    """
    segments = prediction_segments(text)

    feature_matrix = np.vstack([
        extract_features(segment)
        for segment in segments
    ])

    probabilities = model.predict_proba(feature_matrix)
    mean_probability = probabilities.mean(axis=0)

    predicted_label = int(np.argmax(mean_probability))

    return predicted_label, mean_probability


# ============================================================
# 6. 메인 실행
# ============================================================

def main() -> None:
    ciphertext1_path = Path("ciphertexts1.txt")
    ciphertext2_path = Path("ciphertexts2.txt")

    ciphertext1 = load_ciphertext(ciphertext1_path)
    ciphertext2 = load_ciphertext(ciphertext2_path)

    print("=" * 70)
    print("원본 암호문 통계")
    print("=" * 70)

    print(
        f"ciphertexts1.txt: 길이={len(ciphertext1)}, "
        f"IC={index_of_coincidence(ciphertext1):.6f}"
    )

    print(
        f"ciphertexts2.txt: 길이={len(ciphertext2)}, "
        f"IC={index_of_coincidence(ciphertext2):.6f}"
    )

    # 앞 문항의 분석 결과에 따라 라벨 지정
    caesar_text = ciphertext1
    vigenere_text = ciphertext2

    x_train, y_train, x_test, y_test = build_dataset(
        caesar_text=caesar_text,
        vigenere_text=vigenere_text,
        window_size=120,
        stride=30,
    )

    print()
    print("=" * 70)
    print("데이터셋 크기")
    print("=" * 70)
    print(f"학습 데이터: {x_train.shape}")
    print(f"검증 데이터: {x_test.shape}")
    print(f"feature 수: {x_train.shape[1]}")

    model = Pipeline(
        steps=[
            # Random Forest에서는 필수는 아니지만,
            # 다른 모델과의 비교 및 feature 규모 안정화를 위해 유지 가능
            ("scaler", StandardScaler()),

            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=8,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print()
    print("=" * 70)
    print("검증 성능")
    print("=" * 70)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print()

    print("Confusion matrix")
    print(confusion_matrix(y_test, y_pred))
    print()

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Caesar", "Vigenere"],
            digits=4,
            zero_division=0,
        )
    )

    print()
    print("=" * 70)
    print("문제 5 암호문 분류 결과")
    print("=" * 70)

    for name, ciphertext in UNKNOWN_CIPHERTEXTS.items():
        predicted_label, probabilities = predict_ciphertext(
            model,
            ciphertext,
        )

        caesar_probability = probabilities[0]
        vigenere_probability = probabilities[1]

        print(f"\n{name}")
        print(f"길이: {len(clean_text(ciphertext))}")
        print(f"IC: {index_of_coincidence(clean_text(ciphertext)):.6f}")
        print(f"예측 유형: {LABEL_NAMES[predicted_label]}")
        print(f"Caesar 확률: {caesar_probability:.4f}")
        print(f"Vigenere 확률: {vigenere_probability:.4f}")

    # Random Forest feature importance 출력
    classifier = model.named_steps["classifier"]
    importances = classifier.feature_importances_

    top_indices = np.argsort(importances)[::-1][:15]

    feature_names = (
        [f"freq_{letter}" for letter in ALPHABET]
        + [
            "whole_ic",
            "entropy",
            "max_frequency",
            "top3_frequency",
            "frequency_variance",
            "minimum_caesar_chi",
            "second_caesar_chi",
            "caesar_chi_gap",
        ]
        + [f"column_ic_{key_length}" for key_length in range(2, 6)]
        + [f"column_ic_gain_{key_length}" for key_length in range(2, 6)]
        + [f"lag_coincidence_{lag}" for lag in range(1, 6)]
        + [
            "repeat_trigram_ratio",
            "repeat_fourgram_ratio",
            "text_length",
        ]
    )

    print()
    print("=" * 70)
    print("상위 feature importance")
    print("=" * 70)

    for rank, feature_index in enumerate(top_indices, start=1):
        print(
            f"{rank:2d}. "
            f"{feature_names[feature_index]:25s} "
            f"{importances[feature_index]:.6f}"
        )


if __name__ == "__main__":
    main()
