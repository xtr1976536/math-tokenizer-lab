"""Full-parameter GPT-2 fine-tuning with one frozen protocol for all tokenizers."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from math_tokenizer.corpus import ROOT
from .adapters import GPT2_REVISION, METHODS, TokenizerAdapter, build_model, method_slug
from .data import catalog_hashes
from .verify import Verifier


def read_split(folder: Path, split: str):
    rows = []
    for method in ("prim_fwd", "prim_bwd", "prim_ibp", "ode1"):
        path = folder / f"{split}_{method}.jsonl"
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    return rows


def common_coverage(folder: Path, gpt2_path: Path, max_tokens=512):
    """Freeze the same eligible raw examples for every tokenizer."""
    output = {"max_tokens": max_tokens, "kept": {}, "excluded": {}, "sha256": {},
              "catalog_overlap_removed": {}, "heldout_composition_removed": {}}
    normalize = lambda s: "".join(s.split())
    catalog = catalog_hashes(ROOT)
    oracle_path = folder / "oracle_valid_ids.json"
    oracle = json.loads(oracle_path.read_text()) if oracle_path.exists() else None
    for split in ("train", "valid", "test"):
        rows = read_split(folder, split)
        possible = {row["id"] for row in rows}
        compositions = {row["id"] for row in rows if any(
            pattern in text for pattern in (r"e^{\sin", r"\sin{\left(e^")
            for text in (row["input_latex"], row["target_latex"]))}
        possible -= compositions
        output["heldout_composition_removed"][split] = len(compositions)
        if oracle is not None:
            possible &= set(oracle[split])
        if split != "train":
            overlaps = {row["id"] for row in rows if
                        hashlib.sha256(normalize(row["input_latex"]).encode()).hexdigest() in catalog or
                        hashlib.sha256(normalize(row["target_latex"]).encode()).hexdigest() in catalog}
            possible -= overlaps
            output["catalog_overlap_removed"][split] = len(overlaps)
        exclusions = defaultdict(list)
        for method in METHODS:
            adapter = TokenizerAdapter(method, gpt2_path)
            for row in rows:
                prompt, answer = adapter.row(row["task"], row["input_latex"], row["target_latex"])
                if len(prompt) + len(answer) > max_tokens:
                    possible.discard(row["id"])
                    exclusions[method].append(row["id"])
        output["kept"][split] = sorted(possible)
        output["excluded"][split] = {name: len(ids) for name, ids in exclusions.items()}
        output["sha256"][split] = hashlib.sha256("\n".join(sorted(possible)).encode()).hexdigest()
    return output


def task_control(adapter, row):
    return adapter.row(row["task"], row["input_latex"], row["target_latex"])


def examples_for(adapter, rows, allowed):
    return [(row, *task_control(adapter, row)) for row in rows if row["id"] in allowed]


def tensor_batch(examples, pad_id, device):
    width = max(len(prompt) + len(answer) for _, prompt, answer in examples)
    input_ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    attention = torch.zeros_like(input_ids)
    for i, (_row, prompt, answer) in enumerate(examples):
        ids = prompt + answer
        input_ids[i, :len(ids)] = torch.tensor(ids)
        labels[i, len(prompt):len(ids)] = torch.tensor(answer)
        attention[i, :len(ids)] = 1
    return input_ids.to(device), attention.to(device), labels.to(device)


def loss_for(model, examples, adapter, masked_ids, device):
    ids, attention, labels = tensor_batch(examples, adapter.special_ids["PAD"], device)
    logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits[:, :-1, :]
    if masked_ids:
        logits[:, :, masked_ids] = -1e9
    target = labels[:, 1:]
    per_token = F.cross_entropy(logits.transpose(1, 2), target, reduction="none", ignore_index=-100)
    count = target.ne(-100).sum(dim=1).clamp_min(1)
    return (per_token.sum(dim=1) / count).mean()


@torch.no_grad()
def generate_one(model, adapter, prompt, masked_ids, device, max_new_tokens=256, beams=1):
    return generate_batch(model,adapter,[prompt],masked_ids,device,max_new_tokens,beams)[0]


@torch.no_grad()
def generate_batch(model, adapter, prompts, masked_ids, device, max_new_tokens=256, beams=1):
    width = max(len(prompt) for prompt in prompts)
    if width + max_new_tokens > 1024:
        max_new_tokens = 1024-width
    if max_new_tokens < 1:
        raise ValueError("prompt exceeds GPT-2 context")
    ids = torch.full((len(prompts),width),adapter.special_ids["PAD"],dtype=torch.long,device=device)
    attention = torch.zeros_like(ids)
    for row,prompt in enumerate(prompts):
        ids[row,-len(prompt):] = torch.tensor(prompt,device=device)
        attention[row,-len(prompt):] = 1
    out = model.generate(input_ids=ids,attention_mask=attention,max_new_tokens=max_new_tokens,
                         do_sample=False,num_beams=beams,num_return_sequences=beams,
                         pad_token_id=adapter.special_ids["PAD"],
                         eos_token_id=adapter.special_ids["EOS"],
                         bad_words_ids=[[i] for i in masked_ids] if masked_ids else None,use_cache=True)
    result = []
    for row in range(len(prompts)):
        candidates = []
        for sequence in out[row*beams:(row+1)*beams].tolist():
            continuation = sequence[width:]
            stopped = adapter.special_ids["EOS"] in continuation
            if stopped:
                continuation = continuation[:continuation.index(adapter.special_ids["EOS"])]
            try:
                text = adapter.decode(continuation)
                candidates.append({"text":text,"stopped":stopped,"tokens":len(continuation)})
            except Exception as exc:
                candidates.append({"text":"","stopped":stopped,"error":type(exc).__name__})
        result.append(candidates)
    return result


def evaluate(model, adapter, examples, verifier, masked_ids, device, limit=None, beams=1,
             protocol="legacy"):
    model.eval()
    if limit is not None:
        examples = examples[:limit]
    results = []
    batch_size = 16 if beams == 1 else 4
    for offset in range(0,len(examples),batch_size):
        batch = examples[offset:offset+batch_size]
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        elif str(device).startswith("mps"):
            torch.mps.synchronize()
        start = time.perf_counter()
        generated = generate_batch(model,adapter,[prompt for _row,prompt,_answer in batch],
                                   masked_ids,device,beams=beams)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        elif str(device).startswith("mps"):
            torch.mps.synchronize()
        elapsed = time.perf_counter()-start
        for (row,_prompt,_answer),candidates in zip(batch,generated):
            if protocol=="text-v2":
                from .phases import extract_answer
                for candidate in candidates: candidate["extracted_answer"]=extract_answer(candidate["text"])
                verdicts=[verifier.verify(row,candidate["extracted_answer"]) for candidate in candidates]
            else:
                verdicts = [verifier.verify(row, candidate["text"]) if candidate.get("stopped") else "truncated"
                            for candidate in candidates]
            results.append({"id":row["id"],"group":row.get("template_group",row["group"]) if protocol=="text-v2" else row["group"],
                            "equation_group":row["group"],"task":row["task"],
                            "method":row["method"],"predictions":candidates,
                            "verdicts":verdicts,"correct":"correct" in verdicts,
                            "latency_seconds":elapsed/len(batch),"protocol":protocol,
                            "generation_truncated":all(not c.get("stopped") for c in candidates)})
    by_method = defaultdict(list)
    for row in results:
        by_method[row["method"] if row["task"] == "integral" else row["task"]].append(row["correct"])
    scores = {name: sum(values)/len(values) for name, values in by_method.items()}
    scores["integration_macro"] = sum(scores.get(name, 0.0) for name in ("prim_fwd", "prim_bwd", "prim_ibp")) / 3
    scores["selection"] = (scores["integration_macro"] + scores.get("ode_ivp", 0.0)) / 2
    return scores, results


def train_one(method, seed, data_dir, gpt2_path, out_dir, epochs=3, microbatch=2,
              effective_batch=32, max_tokens=512, eval_limit=None, device_name="mps",
              mode="full",prompt_style="legacy"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if microbatch < 1 or effective_batch % microbatch:
        raise ValueError("microbatch must divide effective batch")
    coverage_path = data_dir / f"coverage_{max_tokens}.json"
    if not coverage_path.exists():
        raise FileNotFoundError("freeze common coverage first")
    coverage = json.loads(coverage_path.read_text())
    if coverage["max_tokens"] != max_tokens:
        raise ValueError("coverage length mismatch")
    if prompt_style=="text-v2":
        from .phases import PhaseAdapter,configure_model,parameter_digest,save_checkpoint,PROTOCOL
        if method=="gpt2_native":raise ValueError("native GPT-2 is zero-shot and must not enter training")
        if coverage.get("protocol")!=PROTOCOL:raise ValueError("phase coverage protocol mismatch")
        adapter=PhaseAdapter(method,gpt2_path)
    else:adapter = TokenizerAdapter(method, gpt2_path)
    tokenization_start = time.perf_counter()
    train_rows = read_split(data_dir, "train")
    valid_rows = read_split(data_dir, "valid")
    training = examples_for(adapter, train_rows, set(coverage["kept"]["train"]))
    validation = examples_for(adapter, valid_rows, set(coverage["kept"]["valid"]))
    tokenization_seconds = time.perf_counter() - tokenization_start
    if not training or not validation:
        raise ValueError("empty train/validation set")
    model, embedding_hash = build_model(adapter, gpt2_path, out_dir.parent / "init")
    if prompt_style=="text-v2":model=configure_model(model,mode)
    elif mode!="full":raise ValueError("LoRA requires the shared phase prompt protocol")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    device = torch.device(device_name)
    model.to(device)
    masked_ids = adapter.masked_ids()
    embed = model.get_input_embeddings().weight
    backbone = [p for p in model.parameters() if p is not embed and p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": 2e-5 if mode=="full" else 2e-4},
                                   {"params": [embed], "lr": 2e-4}],
                                  betas=(0.9, 0.999), weight_decay=0.01)
    updates_per_epoch = math.ceil(len(training) / effective_batch)
    total_updates = updates_per_epoch * epochs
    warmup = max(1, math.ceil(total_updates * .05))
    def factor(step):
        if step < warmup:
            return (step+1)/warmup
        progress = (step-warmup)/max(1,total_updates-warmup)
        return .5*(1+math.cos(math.pi*progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [factor, factor])
    verifier = Verifier(ROOT / "tmp/SymbolicMathematics")
    output = out_dir / method_slug(method) / str(seed)
    output.mkdir(parents=True, exist_ok=True)
    config = {"method": method, "seed": seed, "gpt2_revision": GPT2_REVISION,
              "mode":mode,"prompt_style":prompt_style,
              "optimizer":"AdamW","backbone_learning_rate":2e-5 if mode=="full" else 2e-4,
              "embedding_learning_rate":2e-4,"warmup_ratio":.05,"gradient_clip":1.0,
              "embedding_sha256": embedding_hash, "vocab_size": adapter.vocab_size,
              "parameters": sum(p.numel() for p in model.parameters()),
              "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
              "trainable_names":[n for n,p in model.named_parameters() if p.requires_grad],
              "epochs": epochs, "microbatch": microbatch,
              "effective_batch": effective_batch, "max_tokens": max_tokens,
              "train_examples": len(training), "valid_examples": len(validation),
              "train_source_tokens": sum(len(prompt) for _, prompt, _ in training),
              "train_target_tokens": sum(len(answer) for _, _, answer in training),
              "tokenization_seconds": tokenization_seconds,
              "coverage_sha256": coverage["sha256"], "device": device_name}
    if mode=="lora":
        from .phases import LORA
        config["lora"]=LORA
        config["frozen_before_sha256"]=parameter_digest(model,True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    history, best, best_tiebreak, step = [], -1.0, float("inf"), 0
    start_all = time.perf_counter()
    max_allocated, max_driver, padded_tokens = 0, 0, 0
    for epoch in range(epochs):
        model.train()
        train_started=time.perf_counter()
        rng = random.Random(seed + epoch*100000)
        order = list(range(len(training)))
        rng.shuffle(order)
        config.setdefault("epoch_order_sha256",[]).append(hashlib.sha256("\n".join(training[i][0]["id"] for i in order).encode()).hexdigest())
        epoch_loss = 0.0
        for offset in range(0, len(order), effective_batch):
            indices = order[offset:offset+effective_batch]
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(indices), microbatch):
                chunk = [training[j] for j in indices[start:start+microbatch]]
                padded_tokens += len(chunk) * max(len(prompt)+len(answer) for _, prompt, answer in chunk)
                loss = loss_for(model, chunk, adapter, masked_ids, device)
                (loss * len(chunk)/len(indices)).backward()
                epoch_loss += float(loss.detach()) * len(chunk)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if device_name == "mps":
                max_allocated = max(max_allocated, torch.mps.current_allocated_memory())
                max_driver = max(max_driver, torch.mps.driver_allocated_memory())
            if step % 100 == 0:
                print(method, seed, "epoch", epoch+1, "step", step, "loss", round(epoch_loss/max(1,offset+len(indices)),4), flush=True)
        if device_name=="mps":torch.mps.synchronize()
        elif device_name=="cuda":torch.cuda.synchronize()
        train_seconds=time.perf_counter()-train_started
        evaluation_started=time.perf_counter()
        scores, _ = evaluate(model, adapter, validation, verifier, masked_ids, device, eval_limit,protocol=prompt_style)
        tie_groups = defaultdict(list)
        for item in validation:
            key = item[0]["method"] if item[0]["task"] == "integral" else item[0]["task"]
            tie_groups[key].append(item)
        tie_examples = [item for group in sorted(tie_groups) for item in tie_groups[group][:100]]
        with torch.no_grad():
            tie_total = 0.0
            for offset in range(0,len(tie_examples),4):
                chunk=tie_examples[offset:offset+4]
                tie_total += float(loss_for(model,chunk,adapter,masked_ids,device)) * len(chunk)
            tie_loss = tie_total/len(tie_examples)
        if device_name=="mps":torch.mps.synchronize()
        elif device_name=="cuda":torch.cuda.synchronize()
        evaluation_seconds=time.perf_counter()-evaluation_started
        record = {"epoch": epoch+1, "train_loss_per_example": epoch_loss/len(training),
                  "validation": scores, "validation_tiebreak_loss": tie_loss,
                  "train_seconds":train_seconds,"validation_seconds":evaluation_seconds,
                  "elapsed_seconds": time.perf_counter()-start_all,
                  "padded_training_tokens_cumulative": padded_tokens,
                  "max_sampled_mps_allocated_bytes": max_allocated,
                  "max_sampled_mps_driver_bytes": max_driver}
        history.append(record)
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(method, seed, "validation", record, flush=True)
        if scores["selection"] > best or (scores["selection"] == best and tie_loss < best_tiebreak):
            best, best_tiebreak = scores["selection"], tie_loss
            if prompt_style=="text-v2":save_checkpoint(model,output/"best",mode)
            else:model.save_pretrained(output / "best", safe_serialization=True)
            (output / "best_meta.json").write_text(json.dumps(record, indent=2) + "\n")
    if mode=="lora":
        config["frozen_after_sha256"]=parameter_digest(model,True)
        if config["frozen_after_sha256"]!=config["frozen_before_sha256"]:raise RuntimeError("LoRA modified frozen GPT-2 backbone")
    (output/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--data", type=Path, default=ROOT / "tmp/downstream_gpt2/data")
    parser.add_argument("--gpt2", type=Path, default=ROOT / "tmp/gpt2-607a30d")
    parser.add_argument("--out", type=Path, default=ROOT / "tmp/downstream_gpt2/runs")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--effective-batch", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--freeze-coverage", action="store_true")
    args = parser.parse_args()
    if args.freeze_coverage:
        result = common_coverage(args.data, args.gpt2, args.max_tokens)
        dest = args.data / f"coverage_{args.max_tokens}.json"
        dest.write_text(json.dumps(result, indent=2) + "\n")
        print(dest, {k: len(v) for k,v in result["kept"].items()})
    else:
        print(train_one(args.method, args.seed, args.data, args.gpt2, args.out,
                        args.epochs, args.microbatch, args.effective_batch,
                        args.max_tokens, args.eval_limit, args.device))


if __name__ == "__main__":
    main()
