#!/usr/bin/env bash
# 과제 8 장애 시험: 수집기를 10분 멈췄다가 다시 켠다 (systemd 를 건드리지 않고 프로세스를 일시정지/재개, sudo 불필요)
LOG=~/ts-inference/reports/gap_test.log
snap() { echo "--- $1 $(date "+%F %T")"; curl -s localhost:8080/health; echo; curl -s localhost:8080/spaces | python3 -c "import json,sys; [print(r[\"space\"], \"anomaly=\", r[\"is_anomaly\"]) for r in json.load(sys.stdin)]"; }
{
PID=$(systemctl show -p MainPID --value ts-collector)
snap "정지 전"
echo "수집기 일시정지 PID=$PID $(date "+%F %T")"; kill -STOP "$PID"
sleep 240; snap "정지 4분 후"
sleep 360; snap "정지 10분 후"
kill -CONT "$PID"; echo "수집기 재개 $(date "+%F %T")"
sleep 150; snap "재개 2.5분 후"
curl -s "localhost:8080/alarms?hours=1"; echo
} >> "$LOG" 2>&1
