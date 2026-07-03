#!/usr/bin/env bash
# Put KataGo (b28, TensorRT-RTX FP8, pondering) on CGOS 9x9 as saigo.kb28trt.
# Creds are SOURCED at runtime (never stored here); the runtime cfg carrying the password is written
# to /tmp mode-600 and removed on exit. Reconnect loop + kill.txt graceful stop (finish game, quit).
set -uo pipefail
KT=/home/taro/code/saigo.online/engines/katago-trt
CREDS="${CGOS_CREDS:-/home/taro/code/archive/mushin-go/training-jax/.cgos_env}"
[ -f "$CREDS" ] && source "$CREDS"
: "${CGOS_PASSWORD:?CGOS_PASSWORD not set (source $CREDS)}"
NAME="${CGOS_NAME:-saigo.kb28trt}"
SERVER="${CGOS_SERVER:-yss-aya.com}"; PORT="${CGOS_PORT:-6809}"
WRAP="$KT/cgos/katago-cgos9-gtp.sh"
CLIENT=/home/taro/code/CGOS/client/cgosGtp.tcl
KILL="$KT/cgos/kill.txt"; rm -f "$KILL"
CFG="$(mktemp /tmp/cgos_XXXXXX.cfg)"; chmod 600 "$CFG"; trap 'rm -f "$CFG"' EXIT
cat > "$CFG" <<CFGEOF
%section server
    server $SERVER
    port $PORT
%section player
    name $NAME
    password $CGOS_PASSWORD
    invoke $WRAP
    priority 7
CFGEOF
echo "[cgos] $NAME -> $SERVER:$PORT  (kill: touch $KILL)"
while [ ! -f "$KILL" ]; do
  tclsh8.6 "$CLIENT" -c "$CFG" -k "$KILL"
  [ -f "$KILL" ] && break
  echo "[cgos] client dropped; reconnecting in 15s"; sleep 15
done
echo "[cgos] stopped."
