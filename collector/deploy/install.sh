#!/usr/bin/env bash
# 보드(Jetson, Ubuntu)에 수집기를 상시 서비스로 설치한다.
#   cd ~/ts-collector && bash deploy/install.sh
# 재부팅 후에도 자동으로 시작되고, 죽으면 systemd 가 다시 띄운다.
set -euo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)"
USER_NAME="$(id -un)"
SERVICE=ts-collector

cd "$DIR"
if [ ! -f .env ]; then
  echo "[!] .env 가 없습니다. 먼저: cp .env.example .env 후 비밀번호를 채우세요." >&2
  exit 1
fi

echo "[1/4] 파이썬 가상환경 준비"
if ! python3 -m venv .venv >/dev/null 2>&1; then
  echo "    python3-venv 가 없어 설치합니다 (sudo 비밀번호를 물을 수 있음)"
  rm -rf .venv
  PYVER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-venv "python${PYVER}-venv" || sudo apt-get install -y -qq python3-venv
  python3 -m venv .venv
fi
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt

echo "[2/4] systemd 서비스 파일 설치 (sudo 비밀번호를 물을 수 있음)"
sed -e "s|__USER__|$USER_NAME|g" -e "s|__DIR__|$DIR|g" deploy/ts-collector.service \
  | sudo tee /etc/systemd/system/$SERVICE.service >/dev/null

echo "[3/4] 부팅 시 자동 시작 등록 + 지금 시작"
sudo systemctl daemon-reload
sudo systemctl enable --now $SERVICE

echo "[4/4] 상태 확인"
sleep 3
systemctl --no-pager --lines=5 status $SERVICE || true
echo
echo "완료. 로그 보기:   journalctl -u $SERVICE -f"
echo "      현황 보기:   .venv/bin/python tools/status.py"
