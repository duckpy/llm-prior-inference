import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

def slugify(s):
    s = s.strip().replace("/", "_").replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s.strip("_")

def find_data_files(bbq_root):
    sub_path = bbq_root / "data"
    files = [sorted(sub_path.glob("*.jsonl"))]
    return files

def read_json_records(path):
    records = []
    text = path.read_text(encoding="utf-8")

    for line in text.splitlines():
        line = line.strip()
        records.append(json.loads(line))
    return records

def build_prompt(context, question, ans0, ans1, ans2):
    return ("Context: " + context.strip() + " Question: " + question.strip() + " Options: A. " + ans0.strip() + " B. " + ans1.strip() + " C. " + ans2.strip() + " Answer with only A, B, or C.") 

def get_group_label(example, ans_key):
    info = example.get("answer_info", {}).get(ans_key)
    if isinstance(info, list) and len(info) >= 2:
        return str(info[1])
    if isinstance(info, tuple) and len(info) >= 2:
        return str(info[1])
    return ""

def reduced_mapping(example):
    polarity = str(example.get("question_polarity", "")).lower()
    if polarity not in {"neg", "nonneg"}:
        return None

    stereotyped_groups = example.get("additional_metadata", {}).get("stereotyped_groups", [])
    stereotyped_groups = {str(g) for g in stereotyped_groups}
    if not stereotyped_groups:
        return None

    option_for_ans = {"ans0": "A", "ans1": "B", "ans2": "C"}
    mapping = {}

    for ans_key, opt in option_for_ans.items():
        group = get_group_label(example, ans_key)
        group_lower = group.lower()
        if group_lower == "unknown":
            mapping[opt] = "U"
            continue

        in_stereo = group in stereotyped_groups
        if polarity == "neg":
            mapping[opt] = "S" if in_stereo else "A"
        else:
            mapping[opt] = "A" if in_stereo else "S"

    vals = list(mapping.values())
    if vals.count("S") != 1 or vals.count("A") != 1 or vals.count("U") != 1:
        return None
    return mapping

@dataclass
class Config:
    bbq_root = "./BBQ"
    outdir = "./bbq_results/prepared"

def prepare(cfg):
    bbq_root = Path(cfg.bbq_root).expanduser().resolve()
    data_files = find_data_files(bbq_root)

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_jsonl = outdir / "bbq_mapped.jsonl"

    kept = []
    for path in data_files:
        for ex in read_json_records(path):

            if str(ex.get("context_condition", "")).lower() != "ambig":
                continue
            mapping = reduced_mapping(ex)
            if mapping is None:
                continue

            category = str(ex.get("category", path.stem))

            answers = {"A": ex.get("ans0", ""), "B": ex.get("ans1", ""), "C": ex.get("ans2", "")}
            inverse = {v: k for k, v in mapping.items()}
            rec = {
                "record_id": f"{slugify(category)}__{ex.get('example_id')}",
                "source_file": path.name,
                "example_id": ex.get("example_id"),
                "question_index": ex.get("question_index"),
                "question_polarity": ex.get("question_polarity"),
                "context_condition": ex.get("context_condition"),
                "category": category,
                "subcategory": ex.get("additional_metadata", {}).get("subcategory"),
                "stereotyped_groups": ex.get("additional_metadata", {}).get("stereotyped_groups", []),
                "context": ex.get("context", ""),
                "question": ex.get("question", ""),
                "answers": answers,
                "answer_info": ex.get("answer_info", {}),
                "label": ex.get("label"),
                "option_to_canonical": mapping,
                "canonical_to_option": inverse,
                "prompt": build_prompt(ex.get("context", ""), ex.get("question", ""), answers["A"], answers["B"], answers["C"])
            }
            kept.append(rec)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(kept)} canonical ambiguous examples to {out_jsonl}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bbq-root", default="./BBQ")
    p.add_argument("--outdir", default="./bbq_results/prepared")
    return Config(**vars(p.parse_args()))

if __name__ == "__main__":
    prepare(parse_args())
