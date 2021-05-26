#!/bin/bash

#PBS -l walltime=06:00:00
#PBS -l nodes=1:ppn=4

#PBS -m ae -M fb1n15@soton.ac.uk
#PBS -o ./output_log_of_tasks/
#PBS -e ./error_log_of_tasks/

#Change to directory from which job was submitted
#remember! cd to the directory that contains the file when run this script
cd "$PBS_O_WORKDIR"

module load conda/4.4.0
source activate auction

python3 /home/fb1n15/simulation-resource-allocation-multi-agent-RL/multi_agent_sarsa.py $VAR1 $VAR2 $VAR3 $VAR4 $VAR5 $VAR6 $VAR7
