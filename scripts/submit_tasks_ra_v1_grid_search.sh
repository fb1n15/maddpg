#!/bin/bash
# submit a lot of tasks to the iridis4

# try different resource coefficients
seed=0
rc=0.3  # resource coefficient
number_of_tasks=100000
for alpha in 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7
do
  for beta in 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7
  do
    for auction_type in "first-price" "second-price"
    do
    qsub -v VAR1=$seed,VAR2=$rc,VAR3=$number_of_tasks,VAR4=$alpha,\
VAR5=$beta,VAR6=$auction_type script_ra_v1_grid_search.sh
    done
  done
done

# submit to the test queue
#alpha=0.1
#beta=1
#auction_type="second-price"
#qsub -q test -v VAR1=$seed,VAR2=$rc,VAR3=$number_of_tasks,VAR4=$alpha,\
#VAR5=$beta,VAR6=$auction_type script_ra_v1_grid_search.sh