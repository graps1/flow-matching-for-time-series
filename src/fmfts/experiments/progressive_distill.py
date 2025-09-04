import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter

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


def load_teacher(model_cls, model_kwargs, checkpoint_path):
    """Load teacher model and freeze its parameters."""
    teacher = model_cls(**model_kwargs)
    state = torch.load(checkpoint_path, weights_only=True)
    teacher.load_state_dict(state["model"])
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher


def main():
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Progressive distillation trainer")
    parser.add_argument("experiment", choices=list(experiment2params.keys()))
    parser.add_argument("stage", type=int, help="Current distillation stage")
    parser.add_argument("teacher_checkpoint", help="Path to teacher checkpoint")
    parser.add_argument("output_path", help="Directory where checkpoints and logs are stored")
    parser.add_argument(
        "--model-type",
        choices=["flow", "single_step"],
        default="flow",
        help="Student model type",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of training iterations",
    )
    args = parser.parse_args()

    params = experiment2params[args.experiment]
    modelparams = params[args.model_type]

    training_kwargs = modelparams.get("training_kwargs", {}).copy()
    batch_size = training_kwargs.pop("batch_size", 8)
    if args.batch_size is not None:
        batch_size = args.batch_size

    dataset_train = params["dataset"]["cls"](mode="train", **params["dataset"]["kwargs"])

    teacher = load_teacher(modelparams["cls"], modelparams["model_kwargs"], args.teacher_checkpoint)
    student = modelparams["cls"](teacher_model=teacher, **modelparams["model_kwargs"])

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    writer = SummaryWriter(os.path.join(args.output_path, "runs", f"stage_{args.stage}"))

    for step, loss in enumerate(
        student.train_model(dataset_train, optimizer, batch_size=batch_size, **training_kwargs)
    ):
        writer.add_scalar("loss/train", loss.item(), step)
        if step + 1 >= args.iterations:
            break

    state_dir = os.path.join(args.output_path, "trained_models")
    os.makedirs(state_dir, exist_ok=True)
    torch.save(
        {"model": student.state_dict(), "optimizer": optimizer.state_dict()},
        os.path.join(state_dir, f"stage_{args.stage}.pt"),
    )


if __name__ == "__main__":
    main()