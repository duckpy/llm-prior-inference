import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer

def slugify(s):
    s = s.strip().replace("/", "_").replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s.strip("_")

def parse_model_specs(specs):
    out = {}
    for spec in specs:
        name, path = spec.split("=", 1)
        out[slugify(name)] = path
    return out

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

def load_model(model_path, device):
    local_path = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(local_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(local_path), local_files_only=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.eval()
    return model, tokenizer

def prepare_prompt_ids(tokenizer, prompt_text):
    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
    return encoded["input_ids"].squeeze(0)

def prepare_candidate_ids(tokenizer, candidate_strings):
    ids = []
    for c in candidate_strings:
        x = tokenizer(c, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
        ids.append(x)
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
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

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
            tok_lp = log_probs_vocab[k, pred_positions, :].gather(1, cand.unsqueeze(1)).squeeze(1)
            total_logps.append(tok_lp.sum())

        total_logps = torch.stack(total_logps, dim=0).float()
        normalized_logps = total_logps - torch.logsumexp(total_logps, dim=0)
        probs = torch.exp(normalized_logps)
        return {
            "logp": total_logps.detach().cpu().float(),
            "log_probs_normalized": normalized_logps.detach().cpu().float(),
            "probs": probs.detach().cpu().float(),
            "prompt_ids": prompt_ids.detach().cpu(),
            "candidate_token_ids": [c.detach().cpu() for c in candidate_ids]
        }

def canonical_probs(option_prob, option_to_canonical):
    out = {}
    for opt, canon in option_to_canonical.items():
        out[canon] = float(option_prob[opt])
    return {"S": out["S"], "A": out["A"], "U": out["U"]}

def log_odds_from_canonical_probs(canon_probs, eps):
    ps = max(float(canon_probs["S"]), eps)
    pa = max(float(canon_probs["A"]), eps)
    pu = max(float(canon_probs["U"]), eps)
    return [float(np.log(ps) - np.log(pu)), float(np.log(pa) - np.log(pu))]

@dataclass
class Config:
    prepared_jsonl = "./bbq_results/prepared/bbq_mapped.jsonl"
    outdir = "./bbq_results/scored"
    model_spec = ()
    seed = 0

def score_all(cfg):
    torch.manual_seed(cfg.seed)
    records = load_jsonl(cfg.prepared_jsonl)
    models = parse_model_specs(cfg.model_spec)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_strings = ["A", "B", "C"]

    for model_name, model_path in models.items():
        model_outdir = Path(cfg.outdir) / model_name
        scored_path = model_outdir / "scored_records.jsonl"

        model_outdir.mkdir(parents=True, exist_ok=True)

        model, tokenizer = load_model(model_path, device)

        with scored_path.open("w", encoding="utf-8") as f:
            for rec in records:
                score = score_candidate_continuations(model, tokenizer, rec["prompt"], candidate_strings, device)
                probs = score["probs"].numpy().tolist()
                logps = score["log_probs_normalized"].numpy().tolist()
                option_probs = {"A": probs[0], "B": probs[1], "C": probs[2]}
                option_log_probs = {"A": logps[0], "B": logps[1], "C": logps[2]}
                canon_probs = canonical_probs(option_probs, rec["option_to_canonical"])
                y = log_odds_from_canonical_probs(canon_probs)
                out_rec = dict(rec)
                out_rec.update({"model_name": model_name, "candidate_strings": {"A": candidate_strings[0], "B": candidate_strings[1], "C": candidate_strings[2]}, "option_probs": option_probs, "option_log_probs": option_log_probs, "canonical_probs": canon_probs, "y": y})
                f.write(json.dumps(out_rec) + "\n")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared-jsonl", default="./bbq_results/prepared/bbq_mapped.jsonl")
    p.add_argument("--outdir", default="./bbq_results/scored")
    p.add_argument("--model-spec", action="append", default=[])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    return Config(**vars(args))

if __name__ == "__main__":
    score_all(parse_args())
