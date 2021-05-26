#!/bin/bash
# submit a lot of tasks to the iridis4

# try different resource coefficients
number_of_tasks=1000
rc=0.3  # resource coefficient
high_value_slackness=0
low_value_slackness=6
high_value_proportion=0.2
vc_r=10  # valuation coefficient ratio
r_r=1.2  # resource demand ratio
time_length=$(($number_of_tasks / 40))
n_iterations=2000  # number of steps in hill climbing
step_size_para=0.1  # control the step size in hill climbing
i=0  # seed
for i in $(seq 0 49)
do
    echo "$i"
    for rc in 0.2 0.3 0.4
    do
      qsub -v VAR1="$i",VAR2=$number_of_tasks,VAR3=$rc,VAR4=$high_value_slackness,VAR5=$low_value_slackness,VAR6=$high_value_proportion,VAR7=$vc_r,VAR8=$r_r,VAR9=$time_length,VAR10=$n_iterations,VAR11=$step_size_para script_optimal_pricing_plus.sh
    done
done
