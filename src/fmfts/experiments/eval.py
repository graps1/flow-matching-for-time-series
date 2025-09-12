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


# ---- Helper: optional denormalization for metrics ----
def maybe_denorm(x: torch.Tensor, dataset) -> torch.Tensor:
    try:
        mean = getattr(dataset, "mean", None)
        std = getattr(dataset, "std", None)
        if mean is not None and std is not None:
            return x * std.to(x.device) + mean.to(x.device)
        if hasattr(dataset, "denormalize"):
            xd = dataset.denormalize(x)
            return xd.to(x.device)
    except Exception:
        pass
    return x

# ---- Metrics registry (pluggable) ----
def metric_l2(y, x_pred, x_gt, extra):
    return (x_pred - x_gt).pow(2).mean(dim=list(range(1, x_pred.dim())))  # [B]


def metric_sob(y, x_pred, x_gt, extra):
    from fmfts.utils.loss_fn import sobolev
    alpha = float(extra.get("alpha", 1.0))
    beta = float(extra.get("beta", 0.0))
    # Return per-sample scalars (mean over spatial dims)
    return sobolev(
        x_pred - x_gt,
        alpha=alpha,
        beta=beta,
        t=torch.ones(x_pred.shape[0], device=x_pred.device),
        keepbatch=True,
        pointwise=False,
    )


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
    # Use common.batch_size from the merged eval config
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg["common"]["batch_size"], shuffle=False, num_workers=0)

    # Visualization capture config
    viz_enabled = bool(cfg.get("viz", {}).get("enabled", False))
    viz_n = int(cfg.get("viz", {}).get("n_samples", 0)) if viz_enabled else 0
    viz_channels = list(cfg.get("viz", {}).get("channels", [0, 1])) if viz_enabled else []
    one_step_viz = []

    enabled = metrics_cfg["enabled"]
    metric_specs = {k: METRICS[k] for k in enabled}
    for k in enabled:
        if "params" in metrics_cfg.get("params", {}) and k in metrics_cfg["params"]:
            metric_specs[k] = {**metric_specs[k], "params": {**metric_specs[k].get("params", {}), **metrics_cfg["params"][k]}}

    n_eval = 0
    captured = 0
    for y, x_gt in loader:
        y = y.to(device)
        x_gt = x_gt.to(device)
        b = y.shape[0]

        # Prepare per-batch viz holders (only for the first needed samples)
        local_k = 0
        local_viz = None
        if viz_enabled and captured < viz_n:
            local_k = int(min(viz_n - captured, b))
            local_viz = [{
                "y": y[i].detach().cpu(),
                "gt": x_gt[i].detach().cpu(),
                "preds": {}
            } for i in range(local_k)]

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
                # Optionally denormalize before computing metrics
                x_pred_eval = x_pred
                x_gt_eval = x_gt
                if bool(metrics_cfg.get("use_denormalized", False)):
                    x_pred_eval = maybe_denorm(x_pred_eval, dataset)
                    x_gt_eval = maybe_denorm(x_gt_eval, dataset)
                vals = mspec["fn"](y, x_pred_eval, x_gt_eval, mspec.get("params", {})).detach().cpu().numpy()  # [B]
                for i in range(b):
                    results.append({
                        "mode": "one_step",
                        "model": model_name,
                        "step": None,
                        "metric": mname,
                        "value": float(vals[i])
                    })

            # Capture viz predictions for the first few samples
            if local_k > 0:
                xp = x_pred.detach().cpu()
                for i in range(local_k):
                    local_viz[i]["preds"][model_name] = xp[i]

        # Append local viz for this batch BEFORE early stop so we don't drop snapshots
        if local_k > 0:
            one_step_viz.extend(local_viz)
            captured += local_k
            if captured >= viz_n:
                viz_enabled = False

        n_eval += b
        if n_eval >= cfg["common"]["samples"]:
            break

    return results, one_step_viz


@torch.no_grad()
def evaluate_rollout(models: Dict[str, Any], dataset, cfg, metrics_cfg, device: str):
    results = []
    # Visualization capture config
    viz_enabled = bool(cfg.get("viz", {}).get("enabled", False))
    viz_n = int(cfg.get("viz", {}).get("n_samples", 0)) if viz_enabled else 0
    viz_channels = list(cfg.get("viz", {}).get("channels", [0, 1])) if viz_enabled else []
    # Desired rollout snapshot steps (global target); will be clamped per-sample
    length = cfg["rollout"]["length"]
    viz_steps = cfg.get("viz", {}).get("rollout_steps", None)
    if isinstance(viz_steps, int) and viz_steps >= 2:
        # Evenly spaced steps from 1..length (inclusive)
        sel_steps_global = sorted(set(int(round(s)) for s in np.linspace(1, length, viz_steps)))
    elif isinstance(viz_steps, (list, tuple)) and len(viz_steps) > 0:
        sel_steps_global = sorted(set(int(s) for s in viz_steps if 1 <= int(s) <= length))
        if not sel_steps_global:
            sel_steps_global = [1, max(1, length // 2), length]
    else:
        sel_steps_global = [1, max(1, length // 2), length]
    rollout_viz = []
    enabled = metrics_cfg["enabled"]
    metric_specs = {k: METRICS[k] for k in enabled}
    for k in enabled:
        if "params" in metrics_cfg.get("params", {}) and k in metrics_cfg["params"]:
            metric_specs[k] = {**metric_specs[k], "params": {**metric_specs[k].get("params", {}), **metrics_cfg["params"][k]}}

    latent_policy = cfg["rollout"]["latent_policy"]

    # Helper: infer dataset indices (i1/i2 etc.) from flat sample index for known datasets
    def _infer_indices(ds, flat_idx: int):
        # Default: unknown mapping
        info = {"type": None}
        # NS2D layout: __len__ = n_samples * (T - history - 1)
        if hasattr(ds, "n_samples") and hasattr(ds, "total_sequence_len"):
            i1 = flat_idx % ds.n_samples
            i2 = flat_idx // ds.n_samples
            history = int(getattr(ds, "history", 1))
            start_t = i2 + history  # first GT target index used by __getitem__ for x1
            total_T = int(ds.total_sequence_len)
            info.update({
                "type": "ns2d",
                "i1": int(i1),
                "i2": int(i2),
                "start_t": int(start_t),
                "total_T": int(total_T),
            })
            return info
        # RTI3D Full: idx -> i0 (run), i1 (time)
        if hasattr(ds, "n_runs") and hasattr(ds, "max_time_idx") and hasattr(ds, "get"):
            i0 = flat_idx % ds.n_runs
            t = flat_idx // ds.n_runs
            i1 = t % ds.max_time_idx
            history = int(getattr(ds, "history", 1))
            dt = int(getattr(ds, "dt", 1))
            start_t = i1 + history * dt
            # Total timeline nominally 120 as used in dataset timestamp; fall back to start horizon + max_time_idx
            total_T = int(getattr(ds, "total_sequence_len", 120))
            info.update({
                "type": "rti3d_full",
                "i0": int(i0),
                "i1": int(i1),
                "start_t": int(start_t),
                "total_T": int(total_T),
            })
            return info
        # RTI3D Sliced: idx -> i0 (run), i1 (time), i2 (slice)
        if hasattr(ds, "n_runs") and hasattr(ds, "max_time_idx") and hasattr(ds, "get") and hasattr(ds, "dy"):
            i0 = flat_idx % ds.n_runs
            t = flat_idx // ds.n_runs
            i1 = t % ds.max_time_idx
            i2 = t // ds.max_time_idx
            history = int(getattr(ds, "history", 1))
            dt = int(getattr(ds, "dt", 1))
            start_t = i1 + history * dt
            total_T = int(getattr(ds, "total_sequence_len", 120))
            info.update({
                "type": "rti3d_sliced",
                "i0": int(i0),
                "i1": int(i1),
                "i2": int(i2),
                "start_t": int(start_t),
                "total_T": int(total_T),
            })
            return info
        return info

    # Rollout with batch=1 for clarity
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    n_eval = 0
    for sample_idx, (y0, x1) in enumerate(loader):
        y0 = y0.to(device)
        x1 = x1.to(device)

        # Always infer dataset indices to decouple GT retrieval from visualization
        idx_info = _infer_indices(dataset, sample_idx)
        history = int(getattr(dataset, "history", 1))

        # Determine available horizon and fetch ground-truth rollout sequence when possible
        gt_seq_full = None  # stays on device for metric comparison
        available_len = length
        if idx_info.get("type") == "ns2d":
            total_T = idx_info["total_T"]
            start_t = idx_info["start_t"]
            available_len = max(1, min(length, total_T - start_t))
            try:
                gt_seq_full = dataset.get(idx_info["i1"], start_t, available_len).to(device)
            except Exception:
                gt_seq_full = None
        elif idx_info.get("type") == "rti3d_full":
            # dataset.get(i0, i1, sequence_len)
            total_T = idx_info["total_T"]
            start_t = idx_info["start_t"]
            available_len = max(1, min(length, total_T - start_t))
            try:
                gt_seq_full = dataset.get(idx_info["i0"], start_t, available_len).to(device)
            except Exception:
                gt_seq_full = None
        elif idx_info.get("type") == "rti3d_sliced":
            total_T = idx_info["total_T"]
            start_t = idx_info["start_t"]
            available_len = max(1, min(length, total_T - start_t))
            try:
                gt_seq_full = dataset.get(idx_info["i0"], start_t, idx_info["i2"], available_len).to(device)
            except Exception:
                gt_seq_full = None

        # Prepare viz holder; clamp requested steps to available horizon per sample
        y_curr = y0.clone()
        can_viz = viz_enabled and (len(rollout_viz) < viz_n)
        sample_steps = sorted(set(min(s, available_len) for s in sel_steps_global)) if can_viz else sel_steps_global
        sample_viz = {
            "y0": y0[0].detach().cpu(),
            "gt_step1": x1[0].detach().cpu(),
            "preds": {m: {} for m in models.keys()},
            "gt": {}
        } if can_viz else None
        # If we have a GT sequence, populate viz GT frames
        if sample_viz is not None and gt_seq_full is not None:
            gt_seq_cpu = gt_seq_full.detach().cpu()
            for s in sample_steps:
                if 1 <= s <= gt_seq_cpu.shape[0]:
                    sample_viz["gt"][int(s)] = gt_seq_cpu[s - 1]

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

                # Select ground truth for metrics: prefer sequence GT at this step if configured
                x_gt = x1
                if cfg.get("rollout", {}).get("use_sequence_gt_for_metrics", False) and gt_seq_full is not None:
                    if t < gt_seq_full.shape[0]:
                        x_gt = gt_seq_full[t].unsqueeze(0)

                for mname, mspec in metric_specs.items():
                    # Optionally denormalize before computing metrics
                    x_pred_eval = x_pred
                    x_gt_eval = x_gt
                    if bool(metrics_cfg.get("use_denormalized", False)):
                        x_pred_eval = maybe_denorm(x_pred_eval, dataset)
                        x_gt_eval = maybe_denorm(x_gt_eval, dataset)
                    vals = mspec["fn"](y, x_pred_eval, x_gt_eval, mspec.get("params", {})).detach().cpu().numpy()  # [1]
                    results.append({
                        "mode": "rollout",
                        "model": model_name,
                        "step": int(t + 1),
                        "metric": mname,
                        "value": float(vals[0])
                    })

                # Advance conditioner
                y = x_pred.detach()

                # Capture snapshots for selected (clamped) steps
                if sample_viz is not None and (t + 1) in sample_steps:
                    sample_viz["preds"][model_name][int(t + 1)] = x_pred[0].detach().cpu()

        # Append sample viz BEFORE early stop to avoid dropping last captured sample
        if sample_viz is not None:
            rollout_viz.append(sample_viz)
            if len(rollout_viz) >= viz_n:
                viz_enabled = False

        n_eval += 1
        if n_eval >= cfg["common"]["samples"]:
            break

    return results, rollout_viz


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


def render_pdf(results: List[Dict[str, Any]], outdir: str, cfg_effective: Dict[str, Any], viz: Dict[str, Any] = None, aliases: Dict[str, str] = None):
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

        # One-Step snapshots (optional)
        viz_cfg = cfg_effective.get("viz", {})
        if viz and viz_cfg.get("enabled", False):
            chs = list(viz_cfg.get("channels", [0, 1]))
            models_list = cfg_effective.get("models", [])
            show_cbar = bool(viz_cfg.get("colorbar", False))
            # Limit number of samples
            for si, sample in enumerate(viz.get("one_step", []) or []):
                y = sample.get("y").numpy()
                gt = sample.get("gt").numpy()
                preds = sample.get("preds", {})
                for ch in chs:
                    ncols = 1 + len(models_list)
                    fig, axes = plt.subplots(2, ncols, figsize=(2.5 * ncols, 5.0))
                    # Determine unified color scale across ALL models for this sample/channel
                    data_arrays = [gt[ch]] + [preds[m][ch].numpy() for m in models_list if m in preds]
                    vmin = float(np.min([a.min() for a in data_arrays])) if data_arrays else None
                    vmax = float(np.max([a.max() for a in data_arrays])) if data_arrays else None
                    # Unified error scale across ALL models (errors are non-negative)
                    err_arrays = [np.abs(preds[m][ch].numpy() - gt[ch]) for m in models_list if m in preds]
                    err_vmin = 0.0 if err_arrays else None
                    err_vmax = float(np.max([a.max() for a in err_arrays])) if err_arrays else None
                    # Top row: GT + predictions
                    im = axes[0, 0].imshow(gt[ch], cmap='viridis', vmin=vmin, vmax=vmax)
                    axes[0, 0].set_title(f"GT"); axes[0, 0].axis('off')
                    if show_cbar:
                        fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)
                    for j, m in enumerate(models_list, start=1):
                        if m in preds:
                            im_pred = axes[0, j].imshow(preds[m][ch], cmap='viridis', vmin=vmin, vmax=vmax)
                            if show_cbar:
                                fig.colorbar(im_pred, ax=axes[0, j], fraction=0.046, pad=0.04)
                        axes[0, j].set_title(m)
                        alias = (aliases or {}).get(m)
                        if alias:
                            axes[0, j].text(0.5, -0.15, f"{alias} · ch {ch}", transform=axes[0, j].transAxes, ha='center', va='top', fontsize=8)
                        else:
                            axes[0, j].text(0.5, -0.15, f"ch {ch}", transform=axes[0, j].transAxes, ha='center', va='top', fontsize=8)
                        axes[0, j].axis('off')
                    # Bottom row: error maps |pred-GT|
                    axes[1, 0].axis('off')
                    for j, m in enumerate(models_list, start=1):
                        if m in preds:
                            err = np.abs(preds[m][ch].numpy() - gt[ch])
                            im_err = axes[1, j].imshow(err, cmap='magma', vmin=err_vmin, vmax=err_vmax)
                            if show_cbar:
                                fig.colorbar(im_err, ax=axes[1, j], fraction=0.046, pad=0.04)
                        axes[1, j].set_title(r"$\epsilon_{L2}$"); axes[1, j].axis('off')
                    fig.suptitle(f"One-Step Snapshot #{si} — ch {ch}")
                    pdf.savefig(fig, dpi=dpi); plt.close(fig)

        # Rollout snapshots (optional)
        if viz and viz_cfg.get("enabled", False):
            chs = list(viz_cfg.get("channels", [0, 1]))
            models_list = cfg_effective.get("models", [])
            show_cbar = bool(viz_cfg.get("colorbar", False))
            for si, sample in enumerate(viz.get("rollout", []) or []):
                preds_by_model = sample.get("preds", {})
                gt_by_step = sample.get("gt", {})
                sel_steps = []
                # Collect all recorded steps (sorted)
                for m in models_list:
                    sel_steps.extend(list((preds_by_model.get(m, {}) or {}).keys()))
                sel_steps = sorted(set([int(s) for s in sel_steps]))
                if not sel_steps:
                    continue
                # For each channel, compute unified scales across ALL models and render one figure per channel
                for ch in chs:
                    cols = len(sel_steps)
                    fig, axes = plt.subplots(3, cols, figsize=(2.6 * cols, 7.2))
                    if cols == 1:
                        axes = np.array([axes]).reshape(3, 1)
                    # Unified scales
                    data_arrays = []
                    err_arrays = []
                    for step in sel_steps:
                        if step in gt_by_step:
                            data_arrays.append(gt_by_step[step][ch].numpy())
                        for m in models_list:
                            pm = preds_by_model.get(m, {}) or {}
                            if step in pm:
                                arr = pm[step][ch].numpy()
                                data_arrays.append(arr)
                                if step in gt_by_step:
                                    err_arrays.append(np.abs(arr - gt_by_step[step][ch].numpy()))
                    vmin = float(np.min([a.min() for a in data_arrays])) if data_arrays else None
                    vmax = float(np.max([a.max() for a in data_arrays])) if data_arrays else None
                    err_vmin = 0.0 if err_arrays else None
                    err_vmax = float(np.max([a.max() for a in err_arrays])) if err_arrays else None
                    # Draw GT row
                    for j, step in enumerate(sel_steps):
                        if step in gt_by_step:
                            im_gt = axes[0, j].imshow(gt_by_step[step][ch].numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
                            if show_cbar:
                                fig.colorbar(im_gt, ax=axes[0, j], fraction=0.046, pad=0.04)
                        else:
                            axes[0, j].text(0.5, 0.5, "GT N/A", transform=axes[0, j].transAxes, ha='center', va='center', fontsize=8, color='gray')
                        axes[0, j].set_title(f"t + Δ·{step}")
                        axes[0, j].axis('off')
                    # Draw predictions and errors for the first listed model (to keep one figure per channel)
                    # If you prefer per-model figures, we can iterate models_list and create multiple figures.
                    for j, step in enumerate(sel_steps):
                        for m in models_list:
                            pm = preds_by_model.get(m, {}) or {}
                            if step in pm:
                                im_pred = axes[1, j].imshow(pm[step][ch].numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
                                if show_cbar:
                                    fig.colorbar(im_pred, ax=axes[1, j], fraction=0.046, pad=0.04)
                                break  # show one representative prediction; scales are unified across models
                        axes[1, j].axis('off')
                        # Error row (ε_{L2}) — show max error across models for that step as representative
                        if step in gt_by_step:
                            max_err = None
                            for m in models_list:
                                pm = preds_by_model.get(m, {}) or {}
                                if step in pm:
                                    err = np.abs(pm[step][ch].numpy() - gt_by_step[step][ch].numpy())
                                    max_err = err if max_err is None else np.maximum(max_err, err)
                            if max_err is not None:
                                im_err = axes[2, j].imshow(max_err, cmap='magma', vmin=err_vmin, vmax=err_vmax)
                                if show_cbar:
                                    fig.colorbar(im_err, ax=axes[2, j], fraction=0.046, pad=0.04)
                        axes[2, j].set_title(r"$\epsilon_{L2}$")
                        axes[2, j].axis('off')
                    # Title: explicit mapping of rows to content
                    first_label = models_list[0] if not aliases else (aliases.get(models_list[0], models_list[0]))
                    fig.suptitle(f"Rollout Snapshot #{si} -- ch {ch} -- GT (top row) vs {first_label} (middle row)")
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
    # Defaults for viz if not provided in params
    cfg.setdefault("viz", {"enabled": True, "n_samples": 3, "channels": [0, 1]})
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
    viz_payload = {"one_step": [], "rollout": []}
    modes = [m.strip() for m in args.mode.split(",") if m.strip()]
    if "one_step" in modes:
        rows, viz_os = evaluate_one_step(built, ds_test, {**cfg, **{"one_step": cfg["one_step"], "common": cfg["common"]}}, cfg["metrics"], device)
        results_all.extend(rows)
        viz_payload["one_step"] = viz_os
    if "rollout" in modes:
        rows, viz_ro = evaluate_rollout(built, ds_test, cfg, cfg["metrics"], device)
        results_all.extend(rows)
        viz_payload["rollout"] = viz_ro

    # Outputs
    outdir = args.outdir or os.path.join(args.experiment, "eval", now_tag())
    ensure_dir(outdir)
    csv_path, summ_path = summarize_and_save(results_all, outdir, cfg)
    # Build model aliases from checkpoints for nicer labels
    model_aliases = {}
    if args.checkpoints:
        ckpts = [c.strip() for c in args.checkpoints.split(",")]
        for m, p in zip(models, ckpts):
            base = os.path.basename(p)
            alias = os.path.splitext(base)[0]
            if alias.startswith("state_"):
                alias = alias[len("state_"):]
            model_aliases[m] = alias
    else:
        # Fall back to eval defaults if present
        for m in models:
            p = eval_base.get("models", {}).get(m, {}).get("checkpoint") if isinstance(eval_base, dict) else None
            if p:
                base = os.path.basename(p)
                alias = os.path.splitext(base)[0]
                if alias.startswith("state_"):
                    alias = alias[len("state_"):]
                model_aliases[m] = alias
    pdf_path = render_pdf(results_all, outdir, cfg, viz=viz_payload, aliases=model_aliases)

    print("Saved:", csv_path)
    print("Saved:", summ_path)
    print("Saved:", pdf_path)


if __name__ == "__main__":
    main()
