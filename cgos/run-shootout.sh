#!/usr/bin/env bash
set -u
cd /home/taro/code/saigo.online
BIN=/home/taro/code/saigo.online/engines/katago-trt/build/trt/katago
CFG=/home/taro/code/saigo.online/engines/katago-trt/cgos/shootout.cfg
K9=/home/taro/code/katago/nets/kata9x9-b18c384nbt-20231025.bin.gz
B28=/home/taro/code/katago/nets/kata1-b28c512nbt-s13255194368-d5935380940.bin.gz
B40=/home/taro/code/katago/nets/kata1-b40c768.bin.gz
OV="maxTime=0.4,maxVisits=100000000,ponderingEnabled=false"
OUT=/home/taro/code/saigo.online/engines/katago-trt/cgos/shootout
rm -rf ~/.katago/trtcache* 2>/dev/null
./gauntlet tournament \
  "k9:$BIN gtp -model $K9 -config $CFG -override-config $OV" \
  "b28:$BIN gtp -model $B28 -config $CFG -override-config $OV" \
  "b40:$BIN gtp -model $B40 -config $CFG -override-config $OV" \
  --size 9 --komi 7.5 --komi-stdev 1 -p 4 --movecap 200 --resign \
  -n 60 -j 1 --anchor b28 --out "$OUT" --sgfdir "$OUT/sgf" 2>&1
echo SHOOTOUT_DONE
