#!/bin/bash -l
#PBS -N compute_training_data
#PBS -A UWAS0131
#PBS -l select=1:ncpus=8:mpiprocs=1:mem=80GB
#PBS -l walltime=10:00:00
#PBS -q casper
#PBS -j oe
#PBS -m abe
#PBS -o /glade/work/dstiller/enspred/project/compute_training_data.out

export SRCDIR=/glade/u/home/dstiller/dev/uw-enspred/project

module load conda
conda activate enspred

export PYTHONPATH=$SRCDIR
python ${SRCDIR}/project/scripts/compute_training_data.py
