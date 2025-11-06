#!/bin/sh

EXPERIMENT="dNSE"
MODEL_POSTFIX="NS2D"
# EXPERIMENT="sRTI"
# MODEL_POSTFIX="SlicedRTI3D"
# EXPERIMENT="dRTI"
# MODEL_POSTFIX="FullRTI3D"

# these are the number of iterations for each training setup
ITERS_VELOCITY=200000           # flow matching velocity model
ITERS_DET=200000                # deterministic model
ITERS_RECTIFIER_1=50000         # rectifier stage 1
ITERS_RECTIFIER_2=100000        # rectifier stage 2
ITERS_PD_3=30000                # progressive distillation stage 3 (step size = 2^{-3} = 1/8) <-- this is trained first
ITERS_PD_2=60000                # progressive distillation stage 2 (step size = 2^{-2} = 1/4)
ITERS_PD_1=90000                # progressive distillation stage 1 (step size = 2^{-1} = 1/2)
ITERS_PD_0=120000               # progressive distillation stage 0 (step size = 2^0    = 1)   <-- this is trained last
ITERS_ADD__lmbda_09=170000      # adversarial diffusion distillation with lambda=0.9
ITERS_ADD__lmbda_00=170000      # adversarial diffusion distillation with lambda=0.0, aka Wasserstein GAN
ITERS_DD=100000                 # direct distillation

# COMMENT OUT TO TRAIN RESPECTIVE MODELS


# echo "Training Flow Matching model for $ITERS_VELOCITY iterations"
# python3 trainer.py $EXPERIMENT velocity \
#         --max-iter $ITERS_VELOCITY \
#         # --checkpoint $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt




# echo "Training Deterministic Model for $ITERS_DET iterations"
# python3 trainer.py $EXPERIMENT deterministic \
#         --velocity $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt \
#         --max-iter $ITERS_DET  \
#         # --checkpoint $EXPERIMENT/trained_models/DeterministicModel$MODEL_POSTFIX.pt # <- uncomment to continue training




# echo "Training Rectifier stage=1 for $ITERS_RECTIFIER_1 iterations"
# python3 trainer.py $EXPERIMENT rectifier \
#         --velocity $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt \
#         --max-iter $ITERS_RECTIFIER_1 \
#         # --checkpoint $EXPERIMENT/trained_models/Rectifier__stage_1.pt

# echo "Training Rectifier stage=2 for $ITERS_RECTIFIER_2 iterations"
# python3 trainer.py $EXPERIMENT rectifier \
#         --velocity $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt \
#         --max-iter $ITERS_RECTIFIER_2 \
#         --checkpoint $EXPERIMENT/trained_models/Rectifier__stage_1.pt \
#         --advance # <- this command loads the previous stage and advances to the next stage




# echo "Progressive Distillation stage=3 for $ITERS_PD_3 iterations"
# python3 trainer.py $EXPERIMENT prog_dist \
#         --velocity $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt \
#         --max-iter $ITERS_PD_3  \
#         --pd-stage 3 \
#         --pd-k 2 \
#         # --checkpoint $EXPERIMENT/trained_models/ProgressiveDistillation__k_2__stage_3.pt \

# echo "Progressive Distillation stage=2 for $ITERS_PD_2 iterations"
# python3 trainer.py $EXPERIMENT prog_dist \
#         --velocity $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt \
#         --max-iter $ITERS_PD_2  \
#         --checkpoint $EXPERIMENT/trained_models/ProgressiveDistillation__k_2__stage_3.pt \
#         --advance # <- this command loads the previous stage and advances to the stage 2

# echo "Progressive Distillation stage=1 for $ITERS_PD_1 iterations"
# python3 trainer.py $EXPERIMENT prog_dist \
#         --velocity $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt \
#         --max-iter $ITERS_PD_1  \
#         --checkpoint $EXPERIMENT/trained_models/ProgressiveDistillation__k_2__stage_2.pt \
#         --advance # <- this command loads the previous stage and advances to stage 1

# echo "Progressive Distillation stage=0 for $ITERS_PD_0 iterations"
# python3 trainer.py $EXPERIMENT prog_dist \
#         --velocity $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt \
#         --max-iter $ITERS_PD_0  \
#         --checkpoint $EXPERIMENT/trained_models/ProgressiveDistillation__k_2__stage_1.pt \
#         --advance # <- this command loads the previous stage and advances to stage 0




# echo "Training Adversarial Diffusion model for $ITERS_ADD__lmbda_09 iterations"
# python3 trainer.py $EXPERIMENT add \
#         --velocity $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt \
#         --max-iter $ITERS_ADD__lmbda_09 \
#         --add-lambda 0.9 \
#         # --checkpoint $EXPERIMENT/trained_models/AdversarialDiffusionDistillation__lmbda_0.9.pt  # <- uncomment to continue training

# echo "Training Adversarial Diffusion model for $ITERS_ADD__lmbda_00 iterations"
# python3 trainer.py $EXPERIMENT add \
#         --velocity $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt \
#         --max-iter $ITERS_ADD__lmbda_00 \
#         --add-lambda 0.0 \
#         # --checkpoint $EXPERIMENT/trained_models/AdversarialDiffusionDistillation__lmbda_0.0.pt   # <- uncomment to continue training




# echo "Direct Distillation for $ITERS_DD iterations"
# python3 trainer.py $EXPERIMENT dir_dist \
#         --velocity $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt \
#         --max-iter $ITERS_DD \
#         # --checkpoint $EXPERIMENT/trained_models/DirectDistillationModel$MODEL_POSTFIX.pt

