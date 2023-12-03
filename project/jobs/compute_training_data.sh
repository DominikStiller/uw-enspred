#!/bin/bash -l
#PBS -N compute_training_data
#PBS -A UWAS0131
#PBS -l select=1:ncpus=16:mpiprocs=1:mem=300GB
#PBS -l walltime=24:00:00
#PBS -q casper
#PBS -j oe
#PBS -m abe
#PBS -o /glade/work/dstiller/enspred/project/compute_training_data.out

export SRCDIR=/glade/u/home/dstiller/dev/uw-enspred/project

module load conda
conda activate enspred

export PYTHONPATH=$SRCDIR
python ${SRCDIR}/project/scripts/compute_training_data.py
