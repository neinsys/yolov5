#!/bin/bash
# Sets each CUDA device to persistence mode and sets the application clock
# and power limit to the device's maximum supported values.
# When run with "--dry-run" as first command line argument or not as superuser,
# will display the commands, otherwise it will execute them.
#
# Hint: To run this at boot time, place this script in /root and create a file
# /etc/cron.d/nvidia_boost with the following single line:
# @reboot root /root/nvidia_boost.sh >/dev/null
#
# Author: Jan Schlüter, 2017
# Link : https://gist.github.com/f0k/040b7dc2b726c5522b9b28cb633a8b35

# Handle --help :)
if [[ "$1" == "--help" ]]; then
    head "$0" -n 9 | tail -n +2 | cut -c 3-
    exit
fi

# Handle --dry-run
if [[ $EUID -ne 0 || "$1" == "--dry-run" ]]; then
    echo '# dry-run mode, just showing commands that would be run'
    echo '# (run as root and without --dry-run to execute commands instead)'
    run_or_print=echo
else
    run_or_print=
fi

# Handle the GPUs
num_gpus=$(nvidia-smi -L | wc -l)
for ((i=0; i<num_gpus; i++)); do
    # set persistence mode
    $run_or_print nvidia-smi -i $i -pm 1
    # find maximum supported memory and core clock
    clocks=$(nvidia-smi -i $i -q -d SUPPORTED_CLOCKS | grep -F 'Memory' -A1 | head -n2)
    mem_clock="${clocks#*: }"
    mem_clock="${mem_clock%% MHz*}"
    core_clock="${clocks##*: }"
    core_clock="${core_clock% MHz*}"
    # set application clock to maximum
    if [[ "$mem_clock" != "" && "$core_clock" != "" ]]; then
        $run_or_print nvidia-smi -i $i -ac "$mem_clock,$core_clock"
    fi
    # find maximum supported power limit
    power=$(nvidia-smi -i $i -q -d POWER | grep -F 'Max Power')
    power="${power#*: }"
    power="${power%.00 W}"
    # set power limit to maximum
    if [[ "$power" != "N/A" ]]; then
        $run_or_print nvidia-smi -i $i -pl "$power"
    fi
done
