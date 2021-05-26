#!/bin/bash
# submit a lot of tasks to the iridis4

# try different resource coefficients
num_steps=3000 # number of tasks
alpha=0.02
beta=0.01
num_actions=6
num_trials=100  # number of trials
for rc in 3 4
do
    echo "$rc"
    for epsilon in 1
    do
        qsub -v VAR1=$alpha,VAR2=$beta,VAR3=$epsilon,VAR4=$num_steps,\
VAR5=$num_actions,VAR6=$num_trials,VAR7=$rc script_reverse_auction.sh
    done
done
