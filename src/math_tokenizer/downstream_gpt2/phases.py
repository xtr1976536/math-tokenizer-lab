"""The two user-selected experiments and their common mathematical protocol."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
import torch
from transformers import GPT2LMHeadModel
from .adapters import METHODS,TokenizerAdapter,build_model
from .data import catalog_hashes
from math_tokenizer.corpus import ROOT

PROPOSED=METHODS[:2]
FORMAL=METHODS[:7]
SEEDS=(11,23,37)
PHASES={
    "phase1":{"methods":(*PROPOSED,"gpt2_native"),"trained_methods":PROPOSED,
              "mode":"full","epochs":2,"native_role":"zero_shot_reference"},
    "phase2":{"methods":FORMAL,"trained_methods":FORMAL,"mode":"lora","epochs":2},
}
LORA={"r":8,"lora_alpha":16,"lora_dropout":.05,
      "target_modules":["c_attn","c_proj","c_fc"],"fan_in_fan_out":True,"bias":"none"}
PROTOCOL="shared-text-prompt-first-line-v2"

def prompt_text(task,source):
    instruction={"integral":"Find an antiderivative with respect to x.",
                 "ode_general":"Find a solution of the first-order ODE.",
                 "ode_ivp":"Solve the first-order ODE with the given initial condition."}[task]
    return instruction+" Return only a LaTeX expression.\nProblem: "+source+"\nAnswer: "

def extract_answer(text):
    # No oracle-based candidate selection: always evaluate the first output line.
    first=text.strip().splitlines()[0].strip() if text.strip() else ""
    if first.startswith("Answer:"):first=first[7:].strip()
    first=re.sub(r"^(?:f|F|y)\s*\(\s*x\s*\)\s*=\s*","",first)
    if first.startswith("$") and first.endswith("$"):first=first[1:-1].strip()
    return first

class PhaseAdapter(TokenizerAdapter):
    def __init__(self,name,gpt2_path):
        super().__init__(name,gpt2_path)
        if name=="gpt2_native":
            # The zero-shot reference retains the original vocabulary and weights.
            self.special_ids={"EOS":50256,"PAD":50256,"BOS":50256}
            self.vocab_size=self.base_size
    def row(self,task,source,target):
        return self.encode(prompt_text(task,source)),self.encode(target)+[self.special_ids["EOS"]]
    def masked_ids(self):
        return [] if self.name=="gpt2_native" else super().masked_ids()

def freeze_coverage(data):
    """Tokenizer-independent byte limits give both phases the same eligible IDs."""
    from .train import read_split
    catalog=catalog_hashes(ROOT)
    oracle=json.loads((data/"oracle_valid_ids.json").read_text())
    output={"max_tokens":512,"protocol":PROTOCOL,"kept":{},"sha256":{},"excluded":{}}
    for split in ("train","valid","test"):
        kept=[];reasons={};oracle_ids=set(oracle[split])
        for row in read_split(data,split):
            texts=(row["input_latex"],row["target_latex"])
            reason=None
            if row["id"] not in oracle_ids:reason="oracle_unverified"
            elif any(p in t for p in (r"e^{\sin",r"\sin{\left(e^") for t in texts):reason="heldout_composition"
            elif split!="train" and any(hashlib.sha256("".join(t.split()).encode()).hexdigest() in catalog for t in texts):reason="catalog_overlap"
            elif len(prompt_text(row["task"],texts[0]).encode())+len(texts[1].encode())+1>512 or len(texts[1].encode())>256:reason="shared_byte_limit"
            if reason:reasons[reason]=reasons.get(reason,0)+1
            else:kept.append(row["id"])
        output["kept"][split]=sorted(kept)
        output["sha256"][split]=hashlib.sha256("\n".join(sorted(kept)).encode()).hexdigest()
        output["excluded"][split]=reasons
    (data/"coverage_512.json").write_text(json.dumps(output,indent=2)+"\n")
    return output

def configure_model(model,mode):
    if mode=="full":
        for p in model.parameters():p.requires_grad_(True)
    elif mode=="lora":
        from peft import LoraConfig,get_peft_model
        model=get_peft_model(model,LoraConfig(task_type="CAUSAL_LM",**LORA))
        # New tokenizer IDs require a trained embedding; the output head stays tied.
        model.get_input_embeddings().weight.requires_grad_(True)
    else:raise ValueError(mode)
    if model.get_input_embeddings().weight.data_ptr()!=model.get_output_embeddings().weight.data_ptr():
        raise RuntimeError("input/output embedding tie was broken")
    if mode=="lora":
        unexpected=[n for n,p in model.named_parameters() if p.requires_grad and "lora_" not in n and not n.endswith("wte.weight")]
        if unexpected:raise RuntimeError(f"unexpected trainable backbone parameters: {unexpected}")
    return model

def parameter_digest(model,frozen_only=False):
    digest=hashlib.sha256()
    for name,p in model.named_parameters():
        if frozen_only and p.requires_grad:continue
        digest.update(name.encode());digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def save_checkpoint(model,path,mode):
    path.mkdir(parents=True,exist_ok=True)
    if mode=="full":model.save_pretrained(path,safe_serialization=True)
    else:
        model.save_pretrained(path/"lora",safe_serialization=True,save_embedding_layers=False)
        torch.save(model.get_input_embeddings().weight.detach().cpu(),path/"embeddings.pt")
    (path/"checkpoint_kind.json").write_text(json.dumps({"mode":mode,"protocol":PROTOCOL})+"\n")

def load_checkpoint(method,gpt2,folder,cache,mode):
    if method=="gpt2_native":
        return GPT2LMHeadModel.from_pretrained(gpt2,local_files_only=True)
    kind=json.loads((folder/"checkpoint_kind.json").read_text())
    if kind!={"mode":mode,"protocol":PROTOCOL}:raise RuntimeError("checkpoint adaptation/protocol mismatch")
    if mode=="full":return GPT2LMHeadModel.from_pretrained(folder,local_files_only=True)
    from peft import PeftModel
    model,_=build_model(PhaseAdapter(method,gpt2),gpt2,cache)
    model=PeftModel.from_pretrained(model,folder/"lora",is_trainable=False)
    with torch.no_grad():model.get_input_embeddings().weight.copy_(torch.load(folder/"embeddings.pt",weights_only=True,map_location="cpu"))
    if model.get_input_embeddings().weight.data_ptr()!=model.get_output_embeddings().weight.data_ptr():raise RuntimeError("reload broke tied embedding")
    return model
