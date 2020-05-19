# TensorFlow CNN Benchmarks at ALCF

Author: Huihuo Zheng <huihuo.zheng@anl.gov>

This repo includes some scripts for benchmarking ImageNet models at ALCF. 

1) update to the most recent tf cnn benchmark submodule
   > $ git submodule update --init --recursive

2) checkout the right branch compitable to specific tensorflow version
      
   > $ cd tf_bench
   >
   > $ git checkout cnn_tf_v$1_compatible
   > 
   > $ cd -
   > 
   > $ git clone -b cnn_tf_v$1_compatible tf_bench v$1/tf_bench

3) submission script: qsub.sc
   > $ qsub qsub.sc

