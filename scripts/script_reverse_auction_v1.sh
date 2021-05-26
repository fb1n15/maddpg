#!/bin/bash

#PBS -l walltime=01:00:00
#PBS -l nodes=1:ppn=4

#PBS -m ae -M fb1n15@soton.ac.uk
#PBS -o /home/fb1n15/simulation-resource-allocation-multi-agent-RL/output_log_of_tasks/
#PBS -e /home/fb1n15/simulation-resource-allocation-multi-agent-RL/error_log_of_tasks/

#Change to directory from which job was submitted
#remember! cd to the directory that contains the file when run this script
cd "/home/fb1n15/simulation-resource-allocation-multi-agent-RL/scripts/" || exit

module load conda/4.4.0
source activate auction
module load cplex/12.10
export PYTHONPATH=$PYTHONPATH:/home/fb1n15/simulation-resource-allocation-multi-agent-RL

python3 /home/fb1n15/simulation-resource-allocation-multi-agent-RL/scripts/train_reverse_auction_v1.py $VAR1 $VAR3
python3 /home/fb1n15/simulation-resource-allocation-multi-agent-RL/scripts/simulation_reverse_auction_v1.py $VAR1 $VAR2 $VAR3 $VAR4 $VAR5 $VAR6 $VAR7 $VAR8 $VAR9 $VAR10
