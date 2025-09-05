#!/usr/bin/env python
import os
import json
import time
import argparse
import subprocess

from fmfts.experiments.rti3d_sliced.training_parameters import params as rti3d_sliced_params
from fmfts.experiments.rti3d_full.training_parameters import params as rti3d_full_params
from fmfts.experiments.ns2d.training_parameters import params as ns2d_params
from fmfts.experiments.ks2d.training_parameters import params as ks2d_params


experiment2params = {
    "rti3d_sliced": rti3d_sliced_params,
    "rti3d_full": rti3d_full_params,
    "ns2d": ns2d_params,
    "ks2d": ks2d_params,
}


def as_list(x, length):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(length)]


def main():
    parser = argparse.ArgumentParser(description="Multistage Progressive Distillation Orchestrator")
    parser.add_argument("experiment", choices=list(experiment2params.keys()))
    parser.add_argument("--stages", type=int, default=None, help="Number of PD stages to run (overrides training_parameters if set)")
    parser.add_argument("--stage-iters", type=int, nargs="*", default=None, help="Iterations per stage; single int applies to all stages")
    parser.add_argument("--initial-teacher", default=None, help="Path to initial teacher checkpoint; defaults to training_parameters entry")
    parser.add_argument("--trainer", default="trainer.py", help="Path to single-stage trainer script")
    args = parser.parse_args()

    params = experiment2params[args.experiment]
    mpd = params.get("multistage_pd", {})

    stages = args.stages or mpd.get("stages", 1)
    stage_iters_cfg = args.stage_iters or mpd.get("stage_iters", [params["velocity_pd"].get("training_kwargs", {}).get("max_iters", 10000)])
    stage_iters = as_list(stage_iters_cfg, stages)

    # Resolve teacher path
    default_teacher_name = mpd.get("initial_teacher", "state_velocity_teacher1.pt")
    trained_dir = os.path.join(args.experiment, "trained_models")
    os.makedirs(trained_dir, exist_ok=True)
    teacher_path = args.initial_teacher or os.path.join(trained_dir, default_teacher_name)

    # Manifest for bookkeeping
    manifest_path = os.path.join(args.experiment, "trained_models", "multistage_manifest.json")
    manifest = {
        "experiment": args.experiment,
        "stages": stages,
        "stage_iters": stage_iters,
        "initial_teacher": teacher_path,
        "stages_info": [],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    for s in range(1, stages + 1):
        print(f"[multistage] Stage {s}/{stages} | teacher={teacher_path} | iters={stage_iters[s-1]}")
        # Start a fresh student each stage
        cmd = [
            "python", args.trainer,
            args.experiment, "velocity_pd",
            "--new",
            "--teacher", teacher_path,
            "--max-iters", str(stage_iters[s-1]),
        ]
        start_time = time.time()
        subprocess.run(cmd, check=True)
        end_time = time.time()
        elapsed = end_time - start_time

        # The trainer writes the latest student to this path
        student_ckpt = os.path.join(trained_dir, "state_velocity_pd.pt")
        # Keep a stage-tagged copy for lineage
        stage_student_ckpt = os.path.join(trained_dir, f"stage_{s}_student.pt")
        try:
            import shutil
            shutil.copyfile(student_ckpt, stage_student_ckpt)
        except Exception as e:
            print(f"[multistage] Warning: could not copy {student_ckpt} -> {stage_student_ckpt}: {e}")

        manifest["stages_info"].append({
            "stage": s,
            "teacher": teacher_path,
            "student": student_ckpt,
            "stage_student_copy": stage_student_ckpt,
            "iters": stage_iters[s-1],
            "time_elapsed": elapsed,
        })

        # Promote student to teacher for next stage
        teacher_path = student_ckpt

        # Persist manifest after each stage
        try:
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            print(f"[multistage] Warning: could not write manifest {manifest_path}: {e}")


if __name__ == "__main__":
    main()
