#!/bin/bash

#SBATCH --mail-user=raphael97@zedat.fu-berlin.de
#SBATCH --job-name=ni_1_100_1
#SBATCH --mail-type=end
#SBATCH --mem=4000
#SBATCH --time=30:00:00
#SBATCH --qos=prio

#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

module load torchvision/0.5.0-fosscuda-2019b-Python-3.7.4-PyTorch-1.4.0
module load TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
module load PyTorch/1.4.0-fosscuda-2019b-Python-3.7.4
module load matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4

### Changing to the .py script directory
cd /home/raphael97/ni/theoretical_framework_for_target_propagation

python allCNNC_main.py




# To check your running jobs type:
# squeue -u your_username
