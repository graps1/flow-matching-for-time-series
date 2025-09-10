import os
import json
import argparse
import datetime as dt
from typing import Dict, Any, List

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Reuse experiment params registry from trainer
from fmfts.experiments.trainer import experiment2params


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_tag():
    return dt.datetime.now().isoformat(timespec="seconds").replace(":", "_").replace("-", "_")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


# ---- Metrics registry (pluggable) ----
def metric_l2(y, x_pred, x_gt, extra):
    return (x_pred - x_gt).pow(2).mean(dim=list(range(1, x_pred.dim())))  # [B]


def metric_sob(y, x_pred, x_gt, extra):
    from fmfts.utils.loss_fn import sobolev
    alpha = float(extra.get("alpha", 1.0))
    beta = float(extra.get("beta", 0.0))
    # sobolev returns scalar unless pointwise=True; we want per-sample scalars
    return sobolev(x_pred - x_gt, alpha=alpha, beta=beta, t=torch.ones(x_pred.shape[0], device=x_pred.device))


METRICS = {
    "l2":  {"fn": metric_l2,  "reduce": ["mean", "median", "std"], "scope": "both",    "plot": ["hist", "curve"], "params": {}},
    "sob": {"fn": metric_sob, "reduce": ["mean", "median", "std"], "scope": "both",    "plot": ["hist", "curve"], "params": {"alpha": 1.0, "beta": 0.0}},
}


def reduce_values(vals: np.ndarray, reducers: List[str]) -> Dict[str, float]:
    out = {}
    if "mean" in reducers:
        out["mean"] = float(np.mean(vals))
    if "median" in reducers:
        out["median"] = float(np.median(vals))
    if "std" in reducers:
        out["std"] = float(np.std(vals))
    if "p95" in reducers:
        out["p95"] = float(np.percentile(vals, 95))
    return out


# ---- Model builders (mirror trainer.py patterns) ----
def build_models(exp: str, params: Dict[str, Any], model_names: List[str], ckpt_overrides: Dict[str, str], device: str):
    built = {}
    # Base velocity (for flow/single_step/pd teacher or student construction)
    velocity_cls = params["velocity"]["cls"]
    velocity_kwargs = params["velocity"]["model_kwargs"]

    for name in model_names:
        if name not in params:
            raise ValueError(f"Unknown modeltype '{name}' for experiment '{exp}'")

        model_kwargs = dict(params[name].get("model_kwargs", {}))

        # For flow/single_step/velocity_pd we may need auxiliary models
        if name in ["flow", "single_step", "velocity_pd"]:
            # load student/velocity backbone
            vel = velocity_cls(**velocity_kwargs)
            # resolve velocity checkpoint (teacher1 naming aligns with your trainer)
            vel_ckpt_path = os.path.join(exp, "trained_models", "state_velocity.pt")
            try:
                payload = torch.load(vel_ckpt_path, weights_only=True)
                vel.load_state_dict(payload["model"])
            except Exception as e:
                raise RuntimeError(f"Could not load velocity backbone from {vel_ckpt_path}: {e}")

        if name in ["flow", "single_step"]:
            model_kwargs |= {"velocity_model": vel}

        elif name == "velocity_pd":
            # Build teacher (frozen) for PD constructor
            teacher = velocity_cls(**velocity_kwargs)
            # by default use teacher1 file if present, else fallback to state_velocity.pt
            teacher_path = os.path.join(exp, "trained_models", "state_velocity_teacher1.pt")
            if not os.path.exists(teacher_path):
                teacher_path = os.path.join(exp, "trained_models", "state_velocity.pt")
            tp = torch.load(teacher_path, weights_only=True)
            teacher.load_state_dict(tp["model"])
            model_kwargs |= {"teacher": teacher}

        cls = params[name]["cls"]
        model = cls(**model_kwargs).to(device)

        # Load model state
        ckpt_path = ckpt_overrides.get(name) or os.path.join(exp, "trained_models", f"state_{name}.pt")
        try:
            state = torch.load(ckpt_path, weights_only=True)
            # PD checkpoints are student-only; non-strict load for PD
            strict = name != "velocity_pd"
            model.load_state_dict(state["model"], strict=strict)
        except Exception as e:
            raise RuntimeError(f"Could not load checkpoint for {name} from {ckpt_path}: {e}")

        model.eval()
        built[name] = model

    return built


# ---- Evaluation routines ----
@torch.no_grad()
def evaluate_one_step(models: Dict[str, Any], dataset, cfg, metrics_cfg, device: str):
    results = []  # long-form rows
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    enabled = metrics_cfg["enabled"]
    metric_specs = {k: METRICS[k] for k in enabled}
    for k in enabled:
        if "params" in metrics_cfg.get("params", {}) and k in metrics_cfg["params"]:
            metric_specs[k] = {**metric_specs[k], "params": {**metric_specs[k].get("params", {}), **metrics_cfg["params"][k]}}

    n_eval = 0
    for y, x_gt in loader:
        y = y.to(device)
        x_gt = x_gt.to(device)
        b = y.shape[0]

        for model_name, model in models.items():
            # Resolve steps/method per model if present
            steps = cfg["one_step"]["steps"].get(model_name, 1)
            method = cfg["one_step"]["method"].get(model_name, "midpoint")

            # Dispatch sampling
            if hasattr(model, "sample"):
                if model_name == "velocity":
                    x_pred = model.sample(y, x0=None, steps=steps, method=method)
                else:
                    x_pred = model.sample(y, x0=None, steps=steps)
            else:
                raise RuntimeError(f"Model {model_name} missing sample()")

            # Metrics
            for mname, mspec in metric_specs.items():
                vals = mspec["fn"](y, x_pred, x_gt, mspec.get("params", {})).detach().cpu().numpy()  # [B]
                for i in range(b):
                    results.append({
                        "mode": "one_step",
                        "model": model_name,
                        "step": None,
                        "metric": mname,
                        "value": float(vals[i])
                    })

        n_eval += b
        if n_eval >= cfg["common"]["samples"]:
            break

    return results


@torch.no_grad()
def evaluate_rollout(models: Dict[str, Any], dataset, cfg, metrics_cfg, device: str):
    results = []
    enabled = metrics_cfg["enabled"]
    metric_specs = {k: METRICS[k] for k in enabled}
    for k in enabled:
        if "params" in metrics_cfg.get("params", {}) and k in metrics_cfg["params"]:
            metric_specs[k] = {**metric_specs[k], "params": {**metric_specs[k].get("params", {}), **metrics_cfg["params"][k]}}

    length = cfg["rollout"]["length"]
    latent_policy = cfg["rollout"]["latent_policy"]

    # Rollout with batch=1 for clarity
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    n_eval = 0
    for y0, x1 in loader:
        y0 = y0.to(device)
        x1 = x1.to(device)

        # We need trajectory ground truth sequence if available; if not, compare only next step
        # Here we conservatively compare one step ahead repeatedly if dataset doesn't expose sequences.
        y_curr = y0.clone()

        for model_name, model in models.items():
            # Prepare latent if needed
            z_fixed = None
            if latent_policy == "fixed_latent" and model_name != "velocity":
                z_fixed = model.p0.sample(y_curr.shape).to(device)

            steps = cfg["rollout"]["steps"].get(model_name, 1)
            method = cfg["rollout"]["method"].get(model_name, "midpoint")

            y = y_curr.clone()
            for t in range(length):
                # Predict next
                if model_name == "velocity":
                    x_pred = model.sample(y, x0=None, steps=steps, method=method)
                else:
                    if latent_policy == "new_latent" or z_fixed is None:
                        x_pred = model.sample(y, x0=None, steps=steps)
                    else:
                        x_pred = model.sample(y, x0=z_fixed, steps=steps)

                # Without sequence GT, compare to one-step GT available (x1)
                x_gt = x1

                for mname, mspec in metric_specs.items():
                    vals = mspec["fn"](y, x_pred, x_gt, mspec.get("params", {})).detach().cpu().numpy()  # [1]
                    results.append({
                        "mode": "rollout",
                        "model": model_name,
                        "step": int(t + 1),
                        "metric": mname,
                        "value": float(vals[0])
                    })

                # Advance conditioner
                y = x_pred.detach()

        n_eval += 1
        if n_eval >= cfg["common"]["samples"]:
            break

    return results


def summarize_and_save(results: List[Dict[str, Any]], outdir: str, cfg_effective: Dict[str, Any]):
    import csv
    ensure_dir(outdir)

    # CSV long-form
    csv_path = os.path.join(outdir, "metrics.csv")
    if results:
        keys = ["mode", "model", "step", "metric", "value"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(results)

    # JSON summary (aggregates)
    summary = {}
    import pandas as pd
    if results:
        df = pd.DataFrame(results)
        # per-model, per-mode, per-metric aggregates
        for (mode, model, metric), g in df.groupby(["mode", "model", "metric"]):
            vals = g["value"].to_numpy()
            summary.setdefault(mode, {}).setdefault(model, {})[metric] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals)),
                "count": int(vals.size)
            }

    summ_path = os.path.join(outdir, "summary.json")
    with open(summ_path, "w") as f:
        json.dump({"summary": summary, "config": cfg_effective}, f, indent=2)

    return csv_path, summ_path


def render_pdf(results: List[Dict[str, Any]], outdir: str, cfg_effective: Dict[str, Any]):
    pdf_cfg = cfg_effective.get("artifacts", {}).get("pdf", {})
    fname = pdf_cfg.get("filename", "eval_report.pdf")
    dpi = int(pdf_cfg.get("dpi", 150))
    max_figs_per_model = int(pdf_cfg.get("max_figs_per_model", 6))
    include_curves = bool(pdf_cfg.get("include_rollout_curves", True))

    path = os.path.join(outdir, fname)
    with PdfPages(path) as pdf:
        # Cover page
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        title = f"Evaluation Report\n{cfg_effective.get('experiment','')}  |  {now_tag()}"
        plt.text(0.05, 0.92, title, fontsize=16, weight='bold', va='top')
        plt.text(0.05, 0.88, f"Models: {', '.join(cfg_effective.get('models', []))}", fontsize=10)
        pdf.savefig(fig, dpi=dpi); plt.close(fig)

        # Aggregates & optional curves
        import pandas as pd
        if results:
            df = pd.DataFrame(results)
            # Aggregates table
            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            plt.axis('off')
            rows = []
            for (mode, model, metric), g in df.groupby(["mode", "model", "metric"]):
                vals = g["value"].to_numpy()
                rows.append([mode, model, metric, np.mean(vals), np.median(vals), np.std(vals), len(vals)])
            cols = ["mode", "model", "metric", "mean", "median", "std", "count"]
            tbl = plt.table(cellText=[[f"{v:.4g}" if isinstance(v, float) else v for v in r] for r in rows], colLabels=cols, loc='center')
            tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.3)
            pdf.savefig(fig, dpi=dpi); plt.close(fig)

            # Rollout curves
            if include_curves and (df["mode"] == "rollout").any():
                for metric in df["metric"].unique():
                    d = df[(df["mode"] == "rollout") & (df["metric"] == metric)]
                    if d.empty:
                        continue
                    fig, ax = plt.subplots(figsize=(7.5, 4.0))
                    for model in d["model"].unique():
                        sub = d[d["model"] == model]
                        if sub["step"].isnull().all():
                            continue
                        # mean over samples per step
                        mean_by_step = sub.groupby("step")["value"].mean()
                        ax.plot(mean_by_step.index, mean_by_step.values, label=model)
                    ax.set_title(f"Rollout error curves ({metric})")
                    ax.set_xlabel("step"); ax.set_ylabel(metric)
                    ax.legend()
                    pdf.savefig(fig, dpi=dpi); plt.close(fig)

        # Appendix: configuration dump
        cfg_text = json.dumps(cfg_effective, indent=2)
        fig = plt.figure(figsize=(8.5, 11)); plt.axis('off')
        plt.text(0.02, 0.98, "Evaluation Configuration", fontsize=12, weight='bold', va='top')
        plt.text(0.02, 0.94, cfg_text, fontsize=8, va='top', family='monospace')
        pdf.savefig(fig, dpi=dpi); plt.close(fig)

    return path


def merge_eval_cfg(base: Dict[str, Any], args: argparse.Namespace, experiment: str, models: List[str]):
    # clone base eval config
    cfg = json.loads(json.dumps(base))
    cfg["experiment"] = experiment
    cfg["models"] = models
    # CLI overrides (minimal set; extend as needed)
    if args.samples is not None:
        cfg["common"]["samples"] = args.samples
    if args.batch_size is not None:
        cfg["common"]["batch_size"] = args.batch_size
    if args.device is not None:
        cfg["common"]["device"] = args.device
    if args.rollout_len is not None:
        cfg["rollout"]["length"] = args.rollout_len
    if args.latent_policy is not None:
        cfg["rollout"]["latent_policy"] = args.latent_policy
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help=f"one of {list(experiment2params.keys())}")
    ap.add_argument("--models", default="velocity,flow,velocity_pd", help="comma-separated modeltypes to evaluate")
    ap.add_argument("--mode", default="one_step,rollout", help="comma-separated modes: one_step,rollout")
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--rollout-len", type=int, default=None)
    ap.add_argument("--latent-policy", default=None, choices=["fixed_latent","deterministic","new_latent"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--checkpoints", default=None, help="comma-separated ckpts aligned to models (optional)")
    args = ap.parse_args()

    assert args.experiment in experiment2params
    params = experiment2params[args.experiment]
    eval_base = params.get("eval", {})
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    cfg = merge_eval_cfg(eval_base, args, args.experiment, models)
    set_seed(cfg["common"]["seed"])
    device = cfg["common"]["device"]
    torch.set_default_device(device)

    # Checkpoint overrides mapping
    ckpt_map = {}
    if args.checkpoints:
        ckpts = [c.strip() for c in args.checkpoints.split(",")]
        if len(ckpts) != len(models):
            raise ValueError("--checkpoints must match --models length")
        ckpt_map = {m: p for m, p in zip(models, ckpts)}

    built = build_models(args.experiment, params, models, ckpt_map, device)

    # Datasets
    ds_test = params["dataset"]["cls"](mode="test", **params["dataset"]["kwargs"])

    # Evaluate
    results_all = []
    modes = [m.strip() for m in args.mode.split(",") if m.strip()]
    if "one_step" in modes:
        results_all.extend(evaluate_one_step(built, ds_test, {**cfg, **{"one_step": cfg["one_step"], "common": cfg["common"]}}, cfg["metrics"], device))
    if "rollout" in modes:
        results_all.extend(evaluate_rollout(built, ds_test, cfg, cfg["metrics"], device))

    # Outputs
    outdir = args.outdir or os.path.join(args.experiment, "eval", now_tag())
    ensure_dir(outdir)
    csv_path, summ_path = summarize_and_save(results_all, outdir, cfg)
    pdf_path = render_pdf(results_all, outdir, cfg)

    print("Saved:", csv_path)
    print("Saved:", summ_path)
    print("Saved:", pdf_path)


if __name__ == "__main__":
    main()

