#!/usr/bin/env python
def run_command(run_args, exe='qe', env={}, machine='theta'):
    if machine=='theta':
        cmd = "aprun "
        for r in run_args:
            cmd = cmd + " %s %s " %(r, run_args[r])
        env_str = " "
        for e in env:
            env_str = env_str + " -e %s=%s" %(e, env[e])
        cmd = cmd + env_str + " %s " %exe 
    else:
        cmd = " mpirun "
        r = ""
        for r in run_args:
            if r=='-n':
                cmd = cmd + " -np %s " %run_args[r]
            if r=='-N':
                cmd = cmd + " -ppn %s " %run_args[r]
            if r=='-j' or '-d':
                continue
            cmd = cmd + " %s %s " %(r, run_args[r])
        env_str = " "
        for e in env:
            env_str = env_str + " %s=%s" %(e, env[e])
        cmd = env_str + cmd +  " %s " %exe 
    return cmd
