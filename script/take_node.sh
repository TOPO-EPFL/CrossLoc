#!/bin/bash
#SBATCH --chdir /home/qyan/TransPose
#SBATCH --nodes 1
#SBATCH --cpus-per-task 8
#SBATCH --ntasks 1
#SBATCH --account topo
#SBATCH --mem 64G
#SBATCH --time 72:00:00
#SBATCH --partition gpu
#SBATCH --qos gpu
#SBATCH --gres gpu:1
#SBATCH --exclusive

module load gcc cmake
source /home/qyan/venvtranspose/bin/activate

cd opencv-build
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/install/lib:$(pwd)/install/lib64
python3 -c "import torch; import dsacstar"

echo start at `date`
python3 /home/qyan/TransPose/script/foo.py
echo finished at `date`
