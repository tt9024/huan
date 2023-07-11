#!/usr/bin/python3

import subprocess
import os
import datetime
import signal
import sys
import time
import traceback
import numpy as np

def printfl(text) :
    try:
        dt = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")
        print(str(dt) + " " + str(text))
        sys.stdout.flush()
        sys.stderr.flush()
    except:
        pass

def is_python_proc_alive(proc) :
    return proc.poll() is None

def get_pid_array(cmd_line, match_exact=True):
    # cmd_line in format of path/cmd param1 param2
    # this matches EXACTLY with all pid that matches from left to right
    # return array of [pid], if not found, return []
    # it runs with pgrep -a -x
    # If match_exact is False, it matches all given cmd_line, 
    # i.e. 'bin/ftrader' matches with 'bin/ftrader ftrader1' and 
    # 'bin/ftrader ftrader2'
    line = cmd_line.strip().split(' ')
    pname = line[0].split('/')[-1]
    pid_array = []
    try :
        match_lines = subprocess.check_output(['pgrep', '-a', '-x', pname]).decode().strip().split('\n')
        for ml in match_lines:
            # format of [pid, path/pname, param1, ... ]
            ml = ml.strip().split(' ')

            if len(ml) - 1 < len(line):
                continue
            if match_exact and len(ml) - 1 != len(line):
                continue;

            # check the given parameters against the potential matches
            match = True
            for i, param in enumerate(line[1:]):
                if (ml[i+2] != param):
                    match = False
                    break
            if match:
                pid_array.append(ml[0])
    except Exception as e:
        pass

    return pid_array

def is_pid_alive(pid) :
    try :
        pid = int(pid)
    except:
        printfl(str(pid) + " not an integer! trying cmd line")
        return len(get_pid_array(pid))>0
    pids = [int(pid0) for pid0 in os.listdir('/proc') if pid0.isdigit()]
    return pid in pids

def kill_python_proc(proc) :
    try :
        while is_python_proc_alive(proc) :
            #printfl ('sending sigterm')
            proc.send_signal(signal.SIGTERM)
            cnt = 3
            time.sleep(1)
            while cnt > 0 and is_python_proc_alive(proc):
                cnt -= 1
                time.sleep(1)
            if is_python_proc_alive(proc) :
                #printfl ('sending sigkill')
                proc.send_signal(signal.SIGKILL)  #kill -9
                time.sleep(1)
    except Exception as e :
        printfl ('Exception in Kill: ' + str(e))
        traceback.print_exc()
        return 0
    return 1

# gracefully kills one python process (proc)
def kill_by_pid(pid) :
    try :
        if is_pid_alive(pid):
            #printfl ("killing pid " + str(pid))
            os.system("kill " + str(pid))
            cnt = 3
            time.sleep(1)
            while cnt > 0 and is_pid_alive(pid):
                cnt -= 1
                time.sleep(1)
        else:
            #printfl (str(pid) + " not alive, not killing")
            return

        if is_pid_alive(pid):
            #printfl ( str(pid)  +' is alive, force killing with -9')
            os.system("kill -9 " + str(pid))
            time.sleep(1)
        if not is_pid_alive(pid):
            pass
            #printfl("PID: "+ str(pid) + " killed")
        else:
            printfl( "ERROR!  " + str(pid) + " still alive, cannot be killed!")
    except Exception as e :
        printfl ('Exception in Kill: ' + str(e))
        traceback.print_exc()

# kills a program by its command line
# either an executable, i.e. bin/floor, 
# or an executable plus parameters, 
# i.e. 'bin/ftrader ftrader1'
# This matches all processes and kill them all
def kill_by_name(cmd_line):
    pid_array = get_pid_array(cmd_line)
    for pid in pid_array:
        kill_by_pid(pid)

