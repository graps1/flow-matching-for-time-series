#!/bin/sh

EXPERIMENT="ns2d"
MODEL_POSTFIX="NS2D"
# EXPERIMENT="rti3d_sliced"
# MODEL_POSTFIX="SlicedRTI3D"
# EXPERIMENT="rti3d_full"
# MODEL_POSTFIX="FullRTI3D"

ITERS_VELOCITY=70000
ITERS_DET=500000
ITERS_RECTIFIER_1=20000
ITERS_RECTIFIER_2=40000
ITERS_PD_3=100000
ITERS_PD_2=200000
ITERS_PD_1=300000
ITERS_ADD__lmbda_09=120000
ITERS_ADD__lmbda_05=35000
ITERS_ADD__lmbda_00=50000
ITERS_DD=40000



# echo "Training Adversarial Diffusion model for $ITERS_ADD__lmbda_09 iterations"
# python3 trainer.py $EXPERIMENT add \
#         --velocity $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt \
#         --checkpoint $EXPERIMENT/trained_models/AdversarialDiffusionDistillation__lmbda_0.9.pt \
#         --max-iter $ITERS_ADD__lmbda_09 \
#         --add-lambda 0.9
# 
# 

# echo "Training Adversarial Diffusion model for $ITERS_ADD__lmbda_00 iterations"
# python3 trainer.py $EXPERIMENT add \
#         --velocity $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt \
#         --max-iter $ITERS_ADD__lmbda_00 \
#         --checkpoint $EXPERIMENT/trained_models/AdversarialDiffusionDistillation__lmbda_0.0.pt 
        # --add-lambda 0.0 \
        # --new

# echo "Training Adversarial Diffusion model for $ITERS_ADD__lmbda_09 iterations"
# python3 trainer.py $EXPERIMENT add \
#         --velocity $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt \
#         --max-iter $ITERS_ADD__lmbda_09 \
#         --checkpoint $EXPERIMENT/trained_models/AdversarialDiffusionDistillation__lmbda_0.9.pt \
#         --add-lambda 0.9

echo "Training Adversarial Diffusion model for $ITERS_ADD__lmbda_00 iterations"
python3 trainer.py $EXPERIMENT add \
        --velocity $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt \
        --max-iter $ITERS_ADD__lmbda_00 \
        --add-lambda 0.0 \
        --new 
        # --checkpoint $EXPERIMENT/trained_models/AdversarialDiffusionDistillation__lmbda_0.0.pt \


# echo "Training Adversarial Diffusion model for $ITERS_ADD__lmbda_05 iterations"
# python3 trainer.py $EXPERIMENT add \
#         --velocity $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt \
#         --max-iter $ITERS_ADD__lmbda_05 \
#         --checkpoint $EXPERIMENT/trained_models/AdversarialDiffusionDistillation__lmbda_0.5.pt \
#         --add-lambda 0.5
        # --new

# 
# echo "Training Adversarial Diffusion model for $ITERS_ADD__lmbda_05 iterations"
# python3 trainer.py $EXPERIMENT add \
#         --velocity $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt \
#         --checkpoint $EXPERIMENT/trained_models/AdversarialDiffusionDistillation__lmbda_0.5.pt \
#         --max-iter $ITERS_ADD__lmbda_05 \
#         --add-lambda 0.5 \

# echo "Training Flow Matching model for $ITERS_VELOCITY iterations"
# python3 trainer.py $EXPERIMENT velocity \
#         --max-iter $ITERS_VELOCITY \
#         --new

# echo "Training Adversarial Diffusion model for $ITERS_ADD__lmbda_05 iterations"
# python3 trainer.py $EXPERIMENT add \
#         --velocity $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt \
#         --max-iter $ITERS_ADD__lmbda_05 \
#         --add-lambda 0.5 \
#         --new

# echo "Training Adversarial Diffusion model for $ITERS_ADD__lmbda_00 iterations"
# python3 trainer.py $EXPERIMENT add \
#         --velocity $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt \
#         --checkpoint $EXPERIMENT/trained_models/AdversarialDiffusionDistillation__lmbda_0.0.pt \
#         --max-iter $ITERS_ADD__lmbda_00 \
#         --add-lambda 0.0



# echo "Training Rectifier stage=1 for $ITERS_RECTIFIER_1 iterations"
# python3 trainer.py $EXPERIMENT rectifier \
#         --velocity $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt \
#         --max-iter $ITERS_RECTIFIER_1 \
#         --new
# 
# 
# echo "Training Rectifier stage=2 for $ITERS_RECTIFIER_2 iterations"
# python3 trainer.py $EXPERIMENT rectifier \
#         --velocity $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt \
#         --max-iter $ITERS_RECTIFIER_2 \
#         --checkpoint $EXPERIMENT/trained_models/Rectifier__stage_1.pt \
#         --advance


# echo "Training Deterministic Model for $ITERS_DET iterations"
# python3 trainer.py $EXPERIMENT deterministic \
#         --velocity $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt \
#         --checkpoint $EXPERIMENT/trained_models/DeterministicModel$MODEL_POSTFIX.pt \
#         --max-iter $ITERS_DET  \
#         # --new

# echo "Progressive Distillation stage=3 for $ITERS_PD_3 iterations"
# python3 trainer.py $EXPERIMENT prog_dist \
#         --velocity $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt \
#         --max-iter $ITERS_PD_3  \
#         --pd-stage 3 \
#         --pd-k 2 \
#         --checkpoint $EXPERIMENT/trained_models/ProgressiveDistillation__k_2__stage_3.pt \
#         # --new
# 
# echo "Progressive Distillation stage=2 for $ITERS_PD_2 iterations"
# python3 trainer.py $EXPERIMENT prog_dist \
#         --velocity $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt \
#         --max-iter $ITERS_PD_2  \
#         --checkpoint $EXPERIMENT/trained_models/ProgressiveDistillation__k_2__stage_3.pt \
#         --advance
# 
# echo "Progressive Distillation stage=1 for $ITERS_PD_1 iterations"
# python3 trainer.py $EXPERIMENT prog_dist \
#         --velocity $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt \
#         --max-iter $ITERS_PD_1  \
#         --checkpoint $EXPERIMENT/trained_models/ProgressiveDistillation__k_2__stage_2.pt \
#         --advance

# echo "Progressive Distillation stage=0 for $ITERS_PD_0 iterations"
# python3 trainer.py $EXPERIMENT prog_dist \
#         --velocity $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt \
#         --max-iter $ITERS_PD_0  \
#         --checkpoint $EXPERIMENT/trained_models/ProgressiveDistillation__k_2__stage_0.pt \
#         # --advance \


# echo "Direct Distillation for $ITERS_DD iterations"
# python3 trainer.py $EXPERIMENT dir_dist \
#         --velocity $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt \
#         --checkpoint $EXPERIMENT/trained_models/DirectDistillationModelFullRTI3D.pt \
#         --max-iter $ITERS_DD  \




# FOR CONTINUING TRAINING

# echo "Training Flow Matching model for $ITERS_VELOCITY iterations"
# python3 trainer.py $EXPERIMENT velocity \
#         --max-iter $ITERS_VELOCITY \
#         --checkpoint $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt

# echo "Progressive Distillation stage=3 for $ITERS_PD_3 iterations"
# python3 trainer.py $EXPERIMENT prog_dist \
#         --velocity $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt \
#         --max-iter $ITERS_PD_3  \
#         --checkpoint $EXPERIMENT/trained_models/ProgressiveDistillation__k_2__stage_3.pt \


# echo "Progressive Distillation stage=2 for $ITERS_PD_2 iterations"
# python3 trainer.py $EXPERIMENT prog_dist \
#         --velocity $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt \
#         --max-iter $ITERS_PD_2  \
#         --checkpoint $EXPERIMENT/trained_models/ProgressiveDistillation__k_2__stage_3.pt \
#         --advance \

# 
# echo "Progressive Distillation stage=1 for $ITERS_PD_1 iterations"
# python3 trainer.py $EXPERIMENT prog_dist \
#         --velocity $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt \
#         --max-iter $ITERS_PD_1  \
#         --checkpoint $EXPERIMENT/trained_models/ProgressiveDistillation__k_2__stage_2.pt \
#         --advance \

# 
# echo "Progressive Distillation stage=0 for $ITERS_PD_0 iterations"
# python3 trainer.py $EXPERIMENT prog_dist \
#         --velocity $EXPERIMENT/trained_models/VelocityModel$MODEL_POSTFIX.pt \
#         --max-iter $ITERS_PD_0  \
#         --checkpoint $EXPERIMENT/trained_models/ProgressiveDistillation__k_2__stage_1.pt \
#         --advance \
