#!/usr/bin/env bash

set -Eeuo pipefail

# ============================================================
# Cipher classifier execution script
#
# Expected repository structure:
# .
# ├── cipher_classifier.py
# ├── ciphertexts1.txt
# ├── ciphertexts2.txt
# └── ex.sh
# ============================================================

# ex.sh가 위치한 디렉터리로 이동한다.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN=""

echo "============================================================"
echo " Caesar / Vigenere Cipher Classifier"
echo "============================================================"
echo "[INFO] Working directory: $SCRIPT_DIR"


# ------------------------------------------------------------
# 1. Python 실행 파일 탐색
# ------------------------------------------------------------

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
else
    echo "[ERROR] Python을 찾을 수 없습니다."
    echo "[ERROR] Vessel AI Workspace에 Python 3가 설치되어 있는지 확인하세요."
    exit 1
fi

echo "[INFO] Python: $PYTHON_BIN"
"$PYTHON_BIN" --version


# ------------------------------------------------------------
# 2. 필수 파일 검사
# ------------------------------------------------------------

REQUIRED_FILES=(
    "cipher_classifier.py"
    "ciphertexts1.txt"
    "ciphertexts2.txt"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "[ERROR] 필수 파일이 없습니다: $SCRIPT_DIR/$file"
        exit 1
    fi

    if [[ ! -s "$file" ]]; then
        echo "[ERROR] 파일이 비어 있습니다: $SCRIPT_DIR/$file"
        exit 1
    fi

    echo "[OK] $file"
done


# ------------------------------------------------------------
# 3. Python 가상환경 생성
# ------------------------------------------------------------

if [[ ! -d "$VENV_DIR" ]]; then
    echo "[INFO] Python 가상환경을 생성합니다: $VENV_DIR"

    if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
        echo "[ERROR] 가상환경 생성에 실패했습니다."
        echo "[ERROR] python3-venv 패키지가 필요한 환경인지 확인하세요."
        exit 1
    fi
else
    echo "[INFO] 기존 가상환경을 사용합니다: $VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "[ERROR] 가상환경 Python을 실행할 수 없습니다: $VENV_PYTHON"
    exit 1
fi


# ------------------------------------------------------------
# 4. 의존성 설치
# ------------------------------------------------------------

echo "[INFO] pip를 업데이트합니다."
"$VENV_PYTHON" -m pip install --upgrade pip

if [[ -f "requirements.txt" ]]; then
    echo "[INFO] requirements.txt를 이용해 패키지를 설치합니다."
    "$VENV_PYTHON" -m pip install -r requirements.txt
else
    echo "[INFO] 필수 패키지를 설치합니다: numpy, scikit-learn"
    "$VENV_PYTHON" -m pip install numpy scikit-learn
fi


# ------------------------------------------------------------
# 5. 입력 파일 간단 검증
# ------------------------------------------------------------

echo "[INFO] 암호문 파일을 검증합니다."

"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import re
import sys

paths = [
    Path("ciphertexts1.txt"),
    Path("ciphertexts2.txt"),
]

for path in paths:
    text = path.read_text(encoding="utf-8")
    cleaned = "".join(re.findall(r"[A-Z]", text.upper()))

    if not cleaned:
        print(f"[ERROR] {path}: 유효한 알파벳 암호문이 없습니다.")
        sys.exit(1)

    print(
        f"[OK] {path}: "
        f"원본 문자 수={len(text):,}, "
        f"정제 후 알파벳 수={len(cleaned):,}"
    )
PY


# ------------------------------------------------------------
# 6. 분류기 실행 및 결과 저장
# ------------------------------------------------------------

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/classifier_result_${TIMESTAMP}.log"

echo
echo "============================================================"
echo "[INFO] cipher_classifier.py 실행"
echo "============================================================"
echo "[INFO] 실행 로그: $LOG_FILE"
echo

# stdout + stderr를 모두 터미널과 로그파일에 동시에 저장
if "$VENV_PYTHON" cipher_classifier.py 2>&1 | tee "$LOG_FILE"; then

    echo
    echo "============================================================"
    echo "[SUCCESS] 분류기가 정상 종료되었습니다."
    echo "[SUCCESS] 실행 결과가 저장되었습니다."
    echo "          -> $LOG_FILE"
    echo "============================================================"

else

    echo
    echo "============================================================"
    echo "[ERROR] 실행 중 오류가 발생했습니다."
    echo "[ERROR] 오류 내용은 아래 로그를 확인하세요."
    echo "        -> $LOG_FILE"
    echo "============================================================"

    exit 1
fi
