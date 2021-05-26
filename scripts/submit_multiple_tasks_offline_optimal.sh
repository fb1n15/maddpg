#!/bin/bash
# submit a lot of tasks to the iridis4

# try different resource coefficients
number_of_tasks=300
rc=0.3  # resource coefficient
high_value_slackness=0
low_value_slackness=6
high_value_proportion=0.1
vc_r=3  # valuation coefficient ratio
r_r=3  # resource demand ratio
# shellcheck disable=SC2004
time_length=$(($number_of_tasks / 10))
for i in $(seq 0 49)
do
    echo "$i"
    for rc in 0.2 0.3 0.4
    do
        qsub -v VAR1="$i",VAR2=$number_of_tasks,VAR3=$rc,VAR4=$high_value_slackness,\
VAR5=$low_value_slackness,VAR6=$high_value_proportion,VAR7=$vc_r,VAR8=$r_r,\
VAR9=$time_length script_offline_optimal.sh
    done
done
