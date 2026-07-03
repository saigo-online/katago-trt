#!/usr/bin/env bash
# GTP wrapper CGOS drives: KataGo b28 on the TensorRT-RTX FP8 backend, strong 9x9 config.
exec /home/taro/code/saigo.online/engines/katago-trt/build/trt/katago gtp \
  -model /home/taro/code/katago/nets/kata1-b28c512nbt-s13255194368-d5935380940.bin.gz \
  -config /home/taro/code/saigo.online/engines/katago-trt/cgos/katago-cgos9-fp8.cfg
