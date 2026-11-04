import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn

def slugify(s):
    s = s.strip().replace("/", "_").replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s.strip("_")

def theta_to_u(theta):
    exp_theta = torch.exp(theta)
    denom = 1.0 + exp_theta.sum(dim=-1, keepdim=True)
    first = exp_theta / denom
    last = 1.0 / denom
    return torch.cat([first, last], dim=-1)

def probs_to_log_odds(probs):
    probs = probs.clamp_min(1e-30)
    return torch.log(probs[..., :2]) - torch.log(probs[..., 2:3])

class LatentPriorModel(nn.Module):
    def __init__(self, dim: int = 2):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        return x + self.theta.unsqueeze(0)

def full_objective(theta, y, noise_std, prior_precision):
    pred = theta.unsqueeze(0).expand_as(y)
    data = 0.5 / (noise_std ** 2) * torch.sum((y - pred) ** 2)
    prior = 0.5 * prior_precision * torch.sum(theta ** 2)
    return data + prior

def train_map(y, lr, epochs, noise_std, prior_precision, device):
    model = LatentPriorModel(dim=y.shape[1]).to(device)
    y_dev = y.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = full_objective(model.theta, y_dev, noise_std, prior_precision)
        loss.backward()
        opt.step()
        hist.append(float(loss.detach().cpu().item()))
    return model.cpu(), hist

def laplace(theta_map, n, noise_std, prior_precision, posterior_samples):
    d = theta_map.numel()
    precision = (n / (noise_std ** 2) + prior_precision) * torch.eye(d)
    cov = torch.linalg.inv(precision)
    mvn = torch.distributions.MultivariateNormal(theta_map.float(), covariance_matrix=cov.float())
    theta_samples = mvn.sample((posterior_samples,))
    u_samples = theta_to_u(theta_samples)
    return {
        "method": "analytic_laplace_linear_gaussian",
        "posterior_mean_theta": theta_map.float().tolist(),
        "posterior_cov_theta": cov.float().tolist(),
        "posterior_mean_u": u_samples.mean(dim=0).float().tolist(),
        "posterior_lower_u": torch.quantile(u_samples, 0.025, dim=0).float().tolist(),
        "posterior_upper_u": torch.quantile(u_samples, 0.975, dim=0).float().tolist()
    }

def categorical_entropy(p, eps):
    p = p.clamp_min(eps)
    return float(-(p * p.log()).sum().item())

def kl_div(p, q, eps):
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return float((p * (p / q).log()).sum().item())

def read_scored_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def group_by_category(records):
    out = {}
    for r in records:
        out.setdefault(str(r["category"]), []).append(r)
    return out

@dataclass
class Config:
    scored_dir = "./bbq_results/scored"
    outdir = "./bbq_results/fitted"
    optimizer = "adam"
    lr = 1e-2
    epochs = 400
    prior_precision = 1.0
    noise_std = 1.0
    posterior_samples = 1000
    seed = 0

def fit_all(cfg):
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scored_root = Path(cfg.scored_dir)
    model_dirs = sorted([p for p in scored_root.iterdir() if p.is_dir() and (p / "scored_records.jsonl").exists()])

    for model_dir in model_dirs:
        model_name = model_dir.name
        records = read_scored_jsonl(model_dir / "scored_records.jsonl")
        grouped = group_by_category(records)
        for category, recs in grouped.items():
            cat_slug = slugify(category)
            outdir = Path(cfg.outdir) / model_name / cat_slug
            outdir.mkdir(parents=True, exist_ok=True)
            y = torch.tensor([r["y"] for r in recs], dtype=torch.float32)
            canon_probs = torch.tensor([[r["canonical_probs"]["S"], r["canonical_probs"]["A"], r["canonical_probs"]["U"]] for r in recs])

            model, hist = train_map(y, cfg.optimizer, cfg.lr, cfg.epochs, cfg.noise_std, cfg.prior_precision, device)
            theta_map = model.theta.detach().cpu().float()
            u_map = theta_to_u(theta_map.unsqueeze(0)).squeeze(0)
            empirical_u = canon_probs.mean(dim=0)
            empirical_u = empirical_u / empirical_u.sum()
            empirical_theta = probs_to_log_odds(empirical_u.unsqueeze(0)).squeeze(0)
            metrics = {
                "theta_l2_to_empirical_mean": float(torch.norm(theta_map - empirical_theta).item()),
                "u_l1_to_empirical_mean": float(torch.sum(torch.abs(u_map - empirical_u)).item()),
                "u_kl_empirical_to_hat": kl_div(empirical_u, u_map),
                "map_entropy": categorical_entropy(u_map),
                "empirical_entropy": categorical_entropy(empirical_u),
                "stereotype_minus_antistereotype": float((u_map[0] - u_map[1]).item()),
                "unknown_mass": float(u_map[2].item())
            }
            summary = {
                "config": asdict(cfg),
                "model_name": model_name,
                "category": category,
                "num_records": len(recs),
                "metrics": metrics,
                "empirical_mean_prior": empirical_u.tolist(),
                "map_prior": u_map.tolist(),
                "empirical_mean_theta": empirical_theta.tolist(),
                "theta_map": theta_map.tolist(),
                "class_order": ["S", "A", "U"]
            }
            lap = laplace(theta_map, len(recs), cfg.noise_std, cfg.prior_precision, cfg.posterior_samples)
            training_history = {
                "loss_history": [float(x) for x in hist],
                "epochs": int(cfg.epochs),
                "optimizer": cfg.optimizer,
                "lr": float(cfg.lr),
                "prior_precision": float(cfg.prior_precision),
                "noise_std": float(cfg.noise_std),
                "num_records": len(recs)
            }
            with (outdir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            with (outdir / "laplace_summary.json").open("w", encoding="utf-8") as f:
                json.dump(lap, f, indent=2)
            with (outdir / "training_history.json").open("w", encoding="utf-8") as f:
                json.dump(training_history, f, indent=2)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scored-dir", default="./bbq_results/scored")
    p.add_argument("--outdir", default="./bbq_results/fitted")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--prior-precision", type=float, default=1.0)
    p.add_argument("--noise-std", type=float, default=1.0)
    p.add_argument("--posterior-samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    return Config(**vars(p.parse_args()))

if __name__ == "__main__":
    fit_all(parse_args())
