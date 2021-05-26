#!/bin/bash
# submit a lot of tasks to the iridis4

# try different resource coefficients
seed=1
num_trials=50
number_of_tasks=1000
rc=3  # resource coefficient
high_value_slackness=0
low_value_slackness=6
high_value_proportion=0.2
vc_r=10  # valuation coefficient ratio
r_r=1.2  # resource demand ratio
time_length=$(($number_of_tasks / 40))
for seed in `seq 0 9`
do
	for rc in 0.2 0.3 0.4
	do
	    qsub -v VAR1=$seed,VAR2=$number_of_tasks,VAR3=$rc,VAR4=$high_value_slackness,\
VAR5=$low_value_slackness,VAR6=$high_value_proportion,VAR7=$vc_r,VAR8=$r_r,\
VAR9=$time_length,VAR10=$num_trials script_reverse_auction_v2.sh
	done
done
