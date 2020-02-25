#!/usr/bin/sh
module load tensorflow-1.15.2
module load horovod-0.19.0-oneccl
cd tf_bench
git checkout cnn_tf_v$1_compatible
cd -
git clone -b cnn_tf_v$1_compatible tf_bench v$1/tf_bench
