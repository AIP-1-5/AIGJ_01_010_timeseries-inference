#!/usr/bin/env bash
# 보드에 추론 서비스를 상시 서비스로 설치:  cd ~/ts-inference && bash deploy/install_service.sh
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
USER_NAME="$(id -un)"
cd "$DIR"
echo "[1/3] 파이썬 환경"
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r service/requirements.txt
echo "[2/3] systemd 등록 (sudo 비밀번호를 물을 수 있음)"
sed -e "s|__USER__|$USER_NAME|g" -e "s|__DIR__|$DIR|g" deploy/ts-inference.service | sudo tee /etc/systemd/system/ts-inference.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now ts-inference
echo "[3/3] 상태"
sleep 5
systemctl --no-pager --lines=5 status ts-inference || true
echo "조회 API: curl -s localhost:8080/spaces | head"
