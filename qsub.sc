#!/usr/bin/sh
#COBALT -n 1 -A datascience -t 1:00:00 -q debug-cache-quad

# I/O profiling -- please check the following page for details 
# https://www.alcf.anl.gov/support-center/theta/darshan-theta

module load datascience/tensorflow-1.15
export DARSHAN_PRELOAD=/soft/perftools/darshan/darshan-3.1.8/lib/libdarshan.so
aprun  -n 1 -N 1 -j 2 -cc depth -e OMP_NUM_THREADS=128 -e LD_PRELOAD=${DARSHAN_PRELOAD} \
    python ./v1.15/tf_bench/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
    --model=alexnet \ # you can change model to inception, resnet, vgg, etc
    --batch_size=32 \
    --num_batches=100 \
    --forward_only=False  \
    --data_format=NCHW \
    --local_parameter_device=cpu \
    --summary_verbosity=1 \
    --num_intra_threads=0 \
    --num_inter_threads=2  \
    --num_warmup_batches=10 \
    --mkl=True \
    --kmp_blocktime=0 \
    --kmp_settings=1 \
    --kmp_affinity="granularity=fine,verbose,compact,1,0" \
    --variable_update=horovod \
    --data_name=imagenet \ 
    --data_dir=/projects/datascience/rzamora/data/imagenet/count.48.size.8m/ 
