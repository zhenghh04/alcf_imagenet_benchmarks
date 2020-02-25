#!/usr/bin/env python
# Tensorflow CNN Benchmarks
import sys, os, time
import numpy.random as rnd
import argparse
from machine import run_command
parser = argparse.ArgumentParser(description='Tensorflow benchmarks')
parser.add_argument('--model', default='alexnet', help='select models')
try:
    parser.add_argument('--num_nodes', default=os.environ["COBALT_JOBSIZE"], type=int, help='number of nodes')
except:
    parser.add_argument('--num_nodes', default=1, type=int, help='number of nodes')
parser.add_argument('--ppn', default=1, type=int, help='number of workers per node')
parser.add_argument('--j', default=2, type=int, help='number of hyperthreads per code')
parser.add_argument('--num_inter', default=2, type=int, help='Number of inter threads')
parser.add_argument('--num_intra', default=0, type=int, help='Number of intra threads')
parser.add_argument('--omp', default=128, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_batches', default=100, type=int)
parser.add_argument('--kmp_blocktime', default=0, type=int)
parser.add_argument('--real', action='store_true')
parser.add_argument('--variable_update', default='horovod')
parser.add_argument('--fusion_threshold', default=None)
parser.add_argument('--version', default='1.10')
parser.add_argument('--jobid', default=None)
parser.add_argument('--directory', default=None)
parser.add_argument('--collective_barrier', action='store_true')
parser.add_argument('--trace_all_events', action='store_true')
parser.add_argument('--kmp_affinity', default='granularity=fine,verbose,compact,1,0')
parser.add_argument('--trace_file', default=None)
parser.add_argument('--cc', default='depth')
parser.add_argument('--vtune', default=None)
parser.add_argument('--kmp_schedule', default=None, help='setting kmp_schedule ')
parser.add_argument('--mkldnn', action='store_true', help='Whether to set MKLDNN_VERBOSE or not')
parser.add_argument('--mkl', action='store_true', help='Whether to set MKL_VERBOSE or not')
parser.add_argument('--root', default='/home/hzheng/datascience/allreduce_benchmark/', help='root directory of the benchmarks')
parser.add_argument('--darshan', action='store_true')
parser.add_argument("--memory_mode", default='Cache', help='memory model [Cache|Flat_HBM|Flat_HBMp|Flat_DDR]')
parser.add_argument("--data_format", default="NCHW")
parser.add_argument("--horovod_timeline", default=None)
parser.add_argument("--mpi", default='craympi', help="Whether to use mpich version or not")
parser.add_argument("--machine", default='theta', help='where to run')
args = parser.parse_args()

run={}
for arg in vars(args):
    print ("%20s: %s" %(arg, getattr(args, arg)))
    run[arg] = getattr(args, arg)
if args.jobid==None:
    jobid="%s%s"%(time.time(), rnd.randint(10000))
else:
    jobid=args.jobid
run['jobid']=jobid
if args.directory == None:
    sdir = jobid
else:
    sdir = args.directory
run['directory'] = sdir
def recMkdir(string):
    directories = string.split('/')
    for d in directories:
        os.system("[ -e %s ] || mkdir %s" %(d, d))
        os.chdir(d)
    for d in directories:
        os.chdir('../')
if sdir[-1]=='/':
    sdir=sdir[:-1]
recMkdir(sdir)
#os.system("[ -e %s ] || mkdir %s" %(sdir, sdir))
if args.omp > 64*args.j/args.ppn:
    args.omp = 64*args.j/args.ppn
if args.real:
    extra=" --data_name=imagenet --data_dir=/projects/datascience/rzamora/data/imagenet/count.48.size.8m/ "
else:
    extra=""
if args.trace_file != None:
    extra = extra + " --trace_file %s " %args.trace_file
def bytes(string):
    if string.find('m')!=-1:
        return int(string[:-1])*1024*1024
    elif string.find('k')!=-1:
        return int(string[:-1])*1024
    elif string.find('g')!=-1:
        return int(string[:-1])*1024*1024*1024
    else:
        return int(string)
env = {'OMP_NUM_THREADS':args.omp}
if args.trace_all_events:
    env['TRACE_ALL_EVENTS']='yes'
if args.collective_barrier:
    env['COLLECTIVE_BARRIER']='yes'
if args.fusion_threshold!=None:
    env['HOROVOD_FUSION_THRESHOLD']=args.fusion_threshold
if args.vtune != None:
    env['LD_LIBRARY_PATH']='$LD_LIBRARY_PATH:/opt/intel/vtune_amplifier/lib64'
    exe= "amplxe-cl -collect %s -r %s -data-limit=10000 ' %(args.vtune, args.vtune)" + exe
if args.kmp_schedule != None:
    env['KMP_SCHEDULE'] = "%s,balanced" %args.kmp_schedule 
    env['OMP_SCHEDULE'] = "%s,balanced" %args.kmp_schedule 
if args.mkldnn:
    env['MKLDNN_VERBOSE'] = 2
if args.mkl:
    env['MKL_VERBOSE'] = 1
if args.horovod_timeline != None:
    env['HOROVOD_TIMELINE'] = args.horovod_timeline

run_args = {}
exe = 'python'
if args.cc.find('unset')==-1:
    run_args['-cc']=args.cc
memory=""
if args.memory_mode.find("Flat_HBMp")!=-1:
    exe = " numactl -p 1 " + exe
elif args.memory_mode.find("Flat_HBM")!=-1:
    exe =" numactl -m 1 " + exe
elif args.memory_mode.find("Flat_DDR")!=-1:
    exe = " numactl -m 0 " + exe
elif args.memory_mode.find("Cache")!=-1:
    exe = exe
if args.darshan:
    env["LD_PRELOAD"] = "$DARSHAN_PRELOAD"
setup=args.version + " "  + args.mpi
run_args = {'-n':args.num_nodes*args.ppn, '-N':args.ppn, '-j':2}
cmd_tmp = run_command=run_command(run_args, env=env, exe=exe, machine=args.machine)


cmd="source %s/setup.sh %s; module list; cd %s; %s %s/v%s/tf_bench/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model=%s --batch_size=%s --num_batches=%s --forward_only=False  --data_format=%s --local_parameter_device=cpu --summary_verbosity=1 --num_intra_threads=%s --num_inter_threads=%s  --num_warmup_batches=10 --mkl=True --kmp_blocktime=%s --kmp_settings=1 --kmp_affinity=%s --variable_update=%s %s |& tee %s.log; cd -" %(args.root, setup, sdir, cmd_tmp, args.root, args.version, args.model, args.batch_size, args.num_batches, args.data_format, args.num_intra, args.num_inter, args.kmp_blocktime, args.kmp_affinity, args.variable_update, extra, jobid)
if args.vtune != None:
    cmd = 'source /opt/intel/vtune_amplifier/amplxe-vars.sh;' + cmd
if (args.darshan):
    cmd = " module load darshan texlive; " + cmd
print("Running the following command: ")
print(" ", cmd)
os.system(cmd)
import json
import numpy as np
os.system("grep 'total images/sec' %s/%s.log | sed -e 's/total images\/sec://g' > %s/%s.dat" %(sdir, jobid, sdir, jobid))
run['images/sec']=np.average(np.loadtxt("%s/%s.dat" %(sdir, jobid)))
data = json.dumps(run)
f = open('%s/%s.json'%(sdir, jobid), 'w')
f.write(data)
f.close()
