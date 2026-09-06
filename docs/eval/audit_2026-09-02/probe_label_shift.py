"""Probe: does the collapsed v5b adapter predict the CURRENT token (identity)
instead of the NEXT token? Compare base / author adapter / v5b checkpoint."""
import os, json, sys, traceback
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
ROOT = "/data/ndloc_bk/NLHV/ntVan/demo"
os.environ["HF_HOME"] = os.path.join(ROOT, "cache/huggingface")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from unsloth import FastLanguageModel

rows = json.load(open(os.path.join(ROOT, "output/train_v5/llm_data_scene_v5.json"), encoding="utf-8"))
samples = rows[:4]

def run(name):
    print(f"\n##### {name}", flush=True)
    model, tok = FastLanguageModel.from_pretrained(model_name=name, max_seq_length=2048,
                                                   load_in_4bit=True, cache_dir=os.path.join(ROOT, "cache"))
    FastLanguageModel.for_inference(model)
    tot = {"same": 0, "next": 0, "n": 0, "ce_shift": 0.0, "ce_unshift": 0.0, "hf_loss": []}
    for r in samples:
        msgs = [{"role": "system", "content": r["instruction"]},
                {"role": "user", "content": r["input"]},
                {"role": "assistant", "content": r["output"]}]
        text = tok.apply_chat_template(msgs, tokenize=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[:, :2048].cuda()
        with torch.no_grad():
            try:
                out = model(input_ids=ids, labels=ids)
                tot["hf_loss"].append(float(out.loss))
            except Exception:
                print("model(labels=) failed:", traceback.format_exc().splitlines()[-1], flush=True)
                out = model(input_ids=ids)
            logits = out.logits.float()
        pred = logits.argmax(-1)[0]
        same = (pred[:-1] == ids[0, :-1]).float().mean().item()   # predicts current token
        nxt = (pred[:-1] == ids[0, 1:]).float().mean().item()     # predicts next token
        ce_shift = torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:]).item()
        ce_unshift = torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, :-1]).item()
        tot["same"] += same; tot["next"] += nxt; tot["n"] += 1
        tot["ce_shift"] += ce_shift; tot["ce_unshift"] += ce_unshift
        print(f"  L={ids.shape[1]} acc_current={same:.3f} acc_next={nxt:.3f} CE_shifted={ce_shift:.3f} CE_unshifted={ce_unshift:.3f}", flush=True)
    n = tot["n"]
    print(f"  MEAN acc_current={tot['same']/n:.3f} acc_next={tot['next']/n:.3f} CE_shifted={tot['ce_shift']/n:.3f} CE_unshifted={tot['ce_unshift']/n:.3f} hf_loss={tot['hf_loss']}", flush=True)
    del model; torch.cuda.empty_cache()

for name in ["unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
             "/data/ndloc_bk/NLHV/ntVan/lora_model_scene_v4_4eps",
             os.path.join(ROOT, "model/refiner_v5b/checkpoint-737")]:
    try:
        run(name)
    except Exception:
        print("FAILED", name, traceback.format_exc(), flush=True)
print("PROBE_DONE", flush=True)
