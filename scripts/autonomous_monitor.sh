#!/bin/bash
# autonomous_monitor.sh
# Her 100 ep'de rapor verir.
# Policy collapse tespit ederse:
#   1. Train'i durdurur
#   2. diagnose_and_fix.py çalıştırır
#   3. Eğitimi sıfırdan yeniden başlatır
#   4. İzlemeye devam eder

PROJ="c:/Users/TRON-V3/OneDrive/Masaüstü/Ders_Dosyalari/Master_Thesis/marl_air_combat"
LOG="$PROJ/logs/train_output.txt"
PYTHON="$PROJ/venv/Scripts/python"

last_ep=0
prev_rew=""
prev_win=""
collapse_streak=0
restart_count=0

echo "[Monitor] Başlatıldı."

while true; do
  sleep 30

  # --- Her 100 ep rapor ---
  for ep in $(seq 100 100 5000); do
    if [ $ep -le $last_ep ]; then continue; fi
    # Tam eşleşme için boşluk duyarlı pattern
    if grep -E "\[Ep +${ep}\]" "$LOG" > /dev/null 2>&1; then
      echo ""
      echo "=== EP ${ep} ==="
      grep -E "\[Ep +${ep}\]|bitiş:|EarlyStop.*Ep ${ep}" "$LOG" | tail -4
      last_ep=$ep
    fi
  done

  # --- Collapse tespiti: son 2 ep satırı ---
  line1=$(grep -E "^\[Ep " "$LOG" 2>/dev/null | tail -2 | head -1)
  line2=$(grep -E "^\[Ep " "$LOG" 2>/dev/null | tail -1)

  if [ -z "$line1" ] || [ -z "$line2" ] || [ "$line1" = "$line2" ]; then
    continue
  fi

  rew1=$(echo "$line1" | grep -oE 'rew=-?[0-9]+\.[0-9]+' | grep -oE '-?[0-9]+\.[0-9]+')
  rew2=$(echo "$line2" | grep -oE 'rew=-?[0-9]+\.[0-9]+' | grep -oE '-?[0-9]+\.[0-9]+')
  win1=$(echo "$line1" | grep -oE 'W=[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+')
  win2=$(echo "$line2" | grep -oE 'W=[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+')

  if [ -z "$rew1" ] || [ -z "$rew2" ] || [ -z "$win1" ] || [ -z "$win2" ]; then
    continue
  fi

  # Python ile numerik karşılaştırma (bash float desteklemez)
  is_collapse=$("$PYTHON" -c "
r1,r2,w1,w2 = $rew1,$rew2,$win1,$win2
print('1' if r1 < -3000 and r2 < -3000 and w1 == 0.0 and w2 == 0.0 else '0')
" 2>/dev/null)

  if [ "$is_collapse" = "1" ]; then
    collapse_streak=$((collapse_streak + 1))
  else
    collapse_streak=0
  fi

  if [ $collapse_streak -ge 2 ]; then
    echo ""
    echo "!!! POLICY COLLAPSE TESPİT EDİLDİ (streak=$collapse_streak) !!!"
    echo "rew=[$rew1, $rew2] win=[$win1, $win2]"

    # 1. Train'i durdur
    ps aux | grep "$PYTHON" | grep "train_mappo" | grep -v grep | awk '{print $1}' | xargs kill 2>/dev/null
    sleep 3
    echo "[Monitor] Train durduruldu."

    # 2. Diagnostik + config fix
    restart_count=$((restart_count + 1))
    echo "[Monitor] Diagnostik çalışıyor (restart #$restart_count)..."
    cd "$PROJ" && "$PYTHON" -X utf8 scripts/diagnose_and_fix.py
    echo "[Monitor] Diagnostik tamamlandı."

    # 3. Log temizle, yeniden başlat
    rm -f "$PROJ/logs/train_output.txt" "$PROJ/logs/train_log.csv"
    sleep 2
    cd "$PROJ" && "$PYTHON" -X utf8 -u training/train_mappo.py --config configs/config.yaml > logs/train_output.txt 2>&1 &
    NEW_PID=$!
    echo "[Monitor] Eğitim yeniden başlatıldı (PID=$NEW_PID)."

    # Reset
    last_ep=0
    collapse_streak=0
    sleep 10

    # Max 3 restart sonra dur
    if [ $restart_count -ge 3 ]; then
      echo "[Monitor] 3 restart denendi, manuel müdahale gerekiyor. Duruyorum."
      break
    fi
  fi

done
