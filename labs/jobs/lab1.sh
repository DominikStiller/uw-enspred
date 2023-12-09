#!/bin/bash -l
#PBS -N lab1
#PBS -A UWAS0131
#PBS -l select=1:ncpus=1:mpiprocs=1:ngpus=1:mem=3GB
#PBS -l gpu_type=v100
#PBS -l walltime=00:15:00
#PBS -q gpudev
#PBS -j oe
#PBS -m abe
#PBS -o /glade/work/dstiller/enspred/lab1/lab1.out

### Load newest CUDA version
module load cuda/11.8

export WORKDIR=/glade/work/dstiller/enspred/lab1/$PBS_JOBID
export SRCDIR=/glade/u/home/dstiller/dev/uw-enspred

mkdir -p $WORKDIR
cd $WORKDIR

### Load your Earth2MIP conda library
module load conda
conda activate earth2mip

### Stuff not working? Uncomment these debugging the runtime env
nvidia-smi
#module list
#conda list
#python -c "import torch; print(torch.cuda.is_available())"

### Run inference
export PYTHONPATH=$SRCDIR
python ${SRCDIR}/enspred/lab1.py
