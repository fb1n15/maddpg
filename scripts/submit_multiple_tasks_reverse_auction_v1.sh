#!/bin/bash
# submit a lot of tasks to the iridis4

# try different resource coefficients
seed=0
num_trials=50
#num_trials=1
number_of_tasks=300
rc=3 # resource coefficient
high_value_slackness=0
low_value_slackness=6
high_value_proportion=0.1
vc_r=3 # valuation coefficient ratio
r_r=3  # resource demand ratio
time_length=$(($number_of_tasks / 10))

for rc in 0.2 0.3 0.4; do
  qsub -v VAR1=$seed,VAR2=$number_of_tasks,VAR3=$rc,VAR4=$high_value_slackness,VAR5=$low_value_slackness,VAR6=$high_value_proportion,VAR7=$vc_r,VAR8=$r_r,VAR9=$time_length,VAR10=$num_trials \
    script_reverse_auction_v1.sh

done

## submit to the test queue
#rc=0.2
#qsub -q test -v VAR1=$seed,VAR2=$number_of_tasks,VAR3=$rc,VAR4=$high_value_slackness,VAR5=$low_value_slackness,VAR6=$high_value_proportion,VAR7=$vc_r,VAR8=$r_r,VAR9=$time_length,VAR10=$num_trials \
#  script_reverse_auction_v1.sh
