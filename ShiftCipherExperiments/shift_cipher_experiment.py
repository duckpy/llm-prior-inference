import argparse
import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from laplace import Laplace

def theta_to_u(theta):
    exp_theta = torch.exp(theta)
    denom = 1.0 + exp_theta.sum(dim=-1, keepdim=True)
    first = exp_theta / denom
    last = 1.0 / denom
    return torch.cat([first, last], dim=-1)

def probs_to_log_odds(probs):
    probs = probs.clamp_min(1e-30)
    return torch.log(probs[..., :-1]) - torch.log(probs[..., -1:])

def random_lowercase_string(length):
    return "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(length))

def build_caesar_prompt(ciphertext):
    return (f"The following text was encrypted using a Caesar cipher: {ciphertext}\n. What is the shift (0-25)? Answer with a single integer.")

def generate_prompts(num_prompts, min_len, max_len):
    prompts = []
    for _ in range(num_prompts):
        L = random.randint(min_len, max_len)
        ciphertext = random_lowercase_string(L)
        prompt_text = build_caesar_prompt(ciphertext)
        prompts.append({"ciphertext": ciphertext, "prompt": prompt_text})
    return prompts

def load_model_and_tokenizer(model_path, device):
    local_path = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(local_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(local_path), local_files_only=True)
    model.to(device)
    model.eval()
    return model, tokenizer

def prepare_prompt_ids(tokenizer, prompt_text):
    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
    return encoded["input_ids"].squeeze(0)

def prepare_candidate_ids(tokenizer, candidate_strings):
    ids = []
    for c in candidate_strings:
        cand_ids = tokenizer(c, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
        ids.append(cand_ids)
    return ids

def score_candidate_continuations(model, tokenizer, prompt_text, candidate_strings, device):
    with torch.no_grad():
        prompt_ids = prepare_prompt_ids(tokenizer, prompt_text)
        prompt_len = int(prompt_ids.numel())

        candidate_ids = prepare_candidate_ids(tokenizer, candidate_strings)
        batch_seqs = []
        cand_lengths = []
        for cand in candidate_ids:
            full_ids = torch.cat([prompt_ids, cand], dim=0)
            batch_seqs.append(full_ids)
            cand_lengths.append(int(cand.numel()))

        max_len = max(seq.numel() for seq in batch_seqs)
        pad_id = tokenizer.pad_token_id

        input_ids = []
        attention_mask = []
        for seq in batch_seqs:
            pad_len = max_len - seq.numel()
            padded = F.pad(seq, (0, pad_len), value=pad_id)
            mask = torch.cat([torch.ones(seq.numel(), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
            input_ids.append(padded)
            attention_mask.append(mask)

        input_ids = torch.stack(input_ids, dim=0).to(device)
        attention_mask = torch.stack(attention_mask, dim=0).to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        log_probs_vocab = torch.log_softmax(logits, dim=-1)

        total_logps = []
        for k, cand in enumerate(candidate_ids):
            cand = cand.to(device)
            cand_len = cand_lengths[k]
            pred_positions = torch.arange(prompt_len - 1, prompt_len + cand_len - 1, device=device)
            token_log_probs = log_probs_vocab[k, pred_positions, :].gather(1, cand.unsqueeze(1)).squeeze(1)
            total_logps.append(token_log_probs.sum())

        total_logps = torch.stack(total_logps, dim=0)
        normalized_logps = total_logps - torch.logsumexp(total_logps, dim=0)
        probs = torch.exp(normalized_logps)

        return {
            "logp": total_logps.detach().cpu(),
            "log_probs_normalized": normalized_logps.detach().cpu(),
            "probs": probs.detach().cpu(),
            "prompt_ids": prompt_ids.detach().cpu(),
            "candidate_token_ids": [c.detach().cpu() for c in candidate_ids]
        }

class ZeroOffsetDataset(Dataset):
    def __init__(self, y):
        self.y = y.float()
        self.x = torch.zeros_like(self.y)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class LatentPriorModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x + self.theta.unsqueeze(0)

def map_loss_regression(pred, target, theta, noise_std, prior_precision):
    data_term = 0.5 / (noise_std ** 2) * torch.sum((target - pred) ** 2)
    prior_term = 0.5 * prior_precision * torch.sum(theta ** 2)
    return data_term + prior_term

def train_map_model(model, train_loader, lr, noise_std, prior_precision, epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    model.to(device)

    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        total_n = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = map_loss_regression(pred, y_batch, model.theta, noise_std, prior_precision)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_n += x_batch.shape[0]

        history.append(total_loss / max(total_n, 1))

    return history

def kl_divergence_categorical(p, q, eps = 1e-12):
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return torch.sum(p * torch.log(p / q)).item()

def summarize_empirical_prior(probs):
    mean_probs = probs.mean(dim=0)
    mean_probs = mean_probs / mean_probs.sum()
    mean_theta = probs_to_log_odds(mean_probs.unsqueeze(0)).squeeze(0)
    return {"mean_probs": mean_probs, "mean_theta": mean_theta}

def compute_reconstruction_metrics(empirical_probs, theta_hat):
    u_hat = theta_to_u(theta_hat.unsqueeze(0)).squeeze(0)
    empirical_mean = empirical_probs.mean(dim=0)
    return {
        "theta_l2_to_empirical_mean": torch.norm(theta_hat - probs_to_log_odds(empirical_mean.unsqueeze(0)).squeeze(0)).item(),
        "u_l1_to_empirical_mean": torch.sum(torch.abs(u_hat - empirical_mean)).item(),
        "u_kl_empirical_to_hat": kl_divergence_categorical(empirical_mean, u_hat)
    }

def fit_laplace_approximation(model, train_loader, hessian_structure = "full"):
    model.eval()
    la = Laplace(model, likelihood="regression", subset_of_weights="all", hessian_structure=hessian_structure)
    la.fit(train_loader)
    return la

def extract_mean_and_cov_from_laplace(la):
    mean = torch.nn.utils.parameters_to_vector(la.model.parameters()).detach().cpu()
    return mean, getattr(la, "posterior_covariance")

@dataclass
class Config:
    model_path = "./models"

    num_classes = 26
    num_prompts = 200
    min_len  = 12
    max_len  = 32

    batch_size = 32
    lr = 1e-2
    epochs = 400
    prior_precision = 1.0
    noise_std = 1.0

    hessian_structure = "full"
    posterior_samples = 1000

    seed = 0
    outdir = "results"

def run_pipeline(cfg):
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    os.makedirs(cfg.outdir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model_and_tokenizer(cfg.model_path, device, cfg.dtype)

    candidate_strings = [str(i) for i in range(cfg.num_classes)]
    tokenization_report = {}
    cand_token_ids = prepare_candidate_ids(tokenizer, candidate_strings)
    for i, ids in enumerate(cand_token_ids):
        tokenization_report[str(i)] = ids.tolist()

    prompts = generate_prompts(cfg.num_prompts, cfg.min_len, cfg.max_len)

    all_probs = []
    for item in prompts:
        result = score_candidate_continuations(model=model, tokenizer=tokenizer, prompt_text=item["prompt"], candidate_strings=candidate_strings, device=device)
        probs = result["probs"]
        all_probs.append(probs)

    probs_tensor = torch.stack(all_probs, dim=0)
    y_tensor = probs_to_log_odds(probs_tensor)

    dataset = ZeroOffsetDataset(y_tensor)
    train_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model_map = LatentPriorModel(dim = cfg.num_classes - 1)
    loss_history = train_map_model(model=model_map, train_loader=train_loader, lr=cfg.lr, noise_std=cfg.noise_std, prior_precision=cfg.prior_precision, epochs=cfg.epochs, device=device)

    theta_map = model_map.theta.detach().cpu()
    u_map = theta_to_u(theta_map.unsqueeze(0)).squeeze(0)

    empirical = summarize_empirical_prior(probs_tensor)
    empirical_u = empirical["mean_probs"]
    empirical_theta = empirical["mean_theta"]
    metrics = compute_reconstruction_metrics(probs_tensor, theta_map)

    summary = {"config": asdict(cfg), "metrics": metrics, "empirical_mean_prior": empirical_u.tolist(), "map_prior": u_map.tolist(), "empirical_mean_theta": empirical_theta.tolist(), "theta_map": theta_map.tolist(), "candidate_tokenization": tokenization_report}
    with open(os.path.join(cfg.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    training_history = {"loss_history": [float(x) for x in loss_history], "epochs": int(cfg.epochs), "optimizer": cfg.optimizer, "lr": float(cfg.lr), "prior_precision": float(cfg.prior_precision), "noise_std": float(cfg.noise_std), "batch_size": int(cfg.batch_size)}
    with open(os.path.join(cfg.outdir, "training_history.json"), "w") as f:
        json.dump(training_history, f, indent=2)

    la = fit_laplace_approximation(model_map.cpu(), train_loader, cfg.hessian_structure)
    post_mean, post_cov = extract_mean_and_cov_from_laplace(la, num_params=cfg.num_classes - 1)

    mvn = torch.distributions.MultivariateNormal(post_mean, covariance_matrix=post_cov)
    theta_samples = mvn.sample((cfg.posterior_samples,))
    u_samples = theta_to_u(theta_samples)
    u_post_mean = u_samples.mean(dim=0)
    u_post_lower = torch.quantile(u_samples, 0.025, dim=0)
    u_post_upper = torch.quantile(u_samples, 0.975, dim=0)
    laplace_result = {"posterior_mean_theta": post_mean.tolist(), "posterior_cov_theta": post_cov.tolist(), "posterior_mean_u": u_post_mean.tolist(), "posterior_lower_u": u_post_lower.tolist(), "posterior_upper_u": u_post_upper.tolist()}
    with open(os.path.join(cfg.outdir, "laplace_summary.json"), "w") as f:
        json.dump(laplace_result, f, indent=2)

def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)

    parser.add_argument("--num-classes", type=int, default=26)
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--min-len", type=int, default=12)
    parser.add_argument("--max-len", type=int, default=32)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--prior-precision", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=1.0)

    parser.add_argument("--hessian-structure", type=str, default="full")
    parser.add_argument("--posterior-samples", type=int, default=1000)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="results")

    args = parser.parse_args()
    return Config(**vars(args))

if __name__ == "__main__":
    run_pipeline(parse_args())