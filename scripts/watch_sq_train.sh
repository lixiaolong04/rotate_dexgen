#!/bin/bash
# 实时监控云端训练 dashboard.
# 用法:  bash scripts/watch_sq_train.sh [刷新秒数, 默认20]
# 退出:  Ctrl-C
#
# 换不同 run 只改下面两行:
#   SEEDS  = 4 个 seed 编号       LOGSUF = 日志后缀 (/root/train_seed<seed>_<LOGSUF>.log)
#   当前: scale-only run = 400-403 / scale   (旧 scale+quat run = 300-303 / sq)
SEEDS="400 401 402 403"
LOGSUF="scale"

HOST="root@180.76.108.184"; PORT=8632
INT=${1:-20}
while true; do
  clear
  echo "===== 训练监控 [seed ${SEEDS} / *_${LOGSUF}.log]  本地 $(date '+%m-%d %H:%M:%S')  (每 ${INT}s 刷新, Ctrl-C 退出) ====="
  ssh -p $PORT -o BatchMode=yes -o ConnectTimeout=8 $HOST "
    echo \"云端时间: \$(date '+%H:%M:%S')\"
    for s in ${SEEDS}; do
      L=/root/train_seed\${s}_${LOGSUF}.log
      if pgrep -f \"seed \$s\" >/dev/null; then st='RUN '
      elif grep -q 'max steps achieved' \"\$L\" 2>/dev/null; then st='DONE'
      else st='DEAD'; fi
      line=\$(grep -E 'Agent Steps:' \"\$L\" 2>/dev/null | tail -1)
      step=\$(echo \"\$line\" | grep -oE 'Agent Steps: [0-9]+M' | grep -oE '[0-9]+M')
      rew=\$(echo  \"\$line\" | grep -oE 'Mean Rewards: [-0-9.]+'  | grep -oE '[-0-9.]+\$')
      best=\$(echo \"\$line\" | grep -oE 'Current Best: [-0-9.]+'  | grep -oE '[-0-9.]+\$')
      gz=\$(grep 'update gravity' \"\$L\" 2>/dev/null | tail -1 | grep -oE '\-?[0-9.]+\)' | tr -d ')')
      ng=\$(grep -c 'update gravity' \"\$L\" 2>/dev/null)
      err=\$(grep -iE 'Traceback|Error:|out of memory|Killed|Segmentation' \"\$L\" 2>/dev/null | head -1)
      printf '  seed %s [%-4s] step=%-7s reward=%-9s best=%-9s 重力%s次 z=%-9s %s\n' \
             \"\$s\" \"\$st\" \"\${step:-..}\" \"\${rew:-..}\" \"\${best:-..}\" \"\${ng:-0}\" \"\${gz:-初始-0.05}\" \"\${err:+!!ERR}\"
    done
    echo '  ---- GPU (idx, mem, util) ----'
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  /'
  " 2>/dev/null || echo "  (SSH 连接失败, 下次刷新重试)"
  sleep $INT
done
