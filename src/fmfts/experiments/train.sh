#!/usr/bin/env bash  
#SBATCH -A NAISS2025-5-327 -p alvis # project name, cluster name, kth mech project :NAISS2023-3-30, NAISS2025-5-327, NAISS2025-5-144
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH -t 4-23:00:00 # time
#SBATCH -J dist_s1
#SBATCH -o tst_rti3d.out

##############################################################################################
#IMP INFO ABOUT ALVIS
# 1. If the job does not need gpu (no cuda) then use : #SBATCH -C NOGPU -n 31  (Cost )
# 2. If the job is memory intensive, use -C flag     : #SBATCH -N 1 --gpus-per-node=V100:2 -C 2xV100   (RAM is ~ 800GB for this option)
#                                                    : #SBATCH -N 1 --gpus-per-node=T4:1 -C MEM1536         (Specify required memory, here 1536 GB atleast)                 
#                                                    : #SBATCH -N 1 --gpus-per-node=A40:1 -C MEM256
#                                                    : #SBATCH -N 1 --gpus-per-node=A100:1 -C MEM512                                                   
#                                                    : #SBATCH -N 1 --gpus-per-node=A100fat:1 -C MEM1024
#
#
#4. Available GPUs and cost/RAM                      : T4:1(0.35/16GB), A40:1(1/40GB), V100:1(1.31), A100:1(1.84), A100fat:1(2.2)                                 
##############################################################################################

module purge
source /mimer/NOBACKUP/groups/kthmech/abhvis/load_modules_v25.sh

# Run DistributedDataParallel with torch.distributed.launch
#python trainer.py ns2d velocity_pd
#python multistage_pd.py ns2d 
python multistage_pd.py rti3d_full

#Test the trained models

#python eval.py ns2d --models velocity_pd,flow,velocity --checkpoints ns2d/trained_models/stage_1_student.pt,ns2d/trained_models/state_flow.pt,ns2d/trained_models/state_velocity_teacher1.pt --mode one_step,rollout #with steps 25,teacher 50
#python eval.py ns2d --models velocity_pd,flow,velocity --checkpoints ns2d/trained_models/stage_2_student.pt,ns2d/trained_models/state_flow.pt,ns2d/trained_models/state_velocity_teacher1.pt --mode one_step,rollout #with steps 12,teacher 50
#python eval.py ns2d --models velocity_pd,flow,velocity --checkpoints ns2d/trained_models/stage_3_student.pt,ns2d/trained_models/state_flow.pt,ns2d/trained_models/state_velocity_teacher1.pt --mode one_step,rollout #with steps 6,teacher 50

