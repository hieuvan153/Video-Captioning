"""Probe 4: kiem tra tinh NHAN QUA cua attention tren duong train (get_peft_model + train mode + attention_mask),
mac dinh vs attn_implementation="eager" (cach tac gia dung). Neu doi token tuong lai lam doi logits o vi tri truoc do
=> mask hong => loss ve 0 la tat yeu (model chep token ke tiep qua attention)."""
import os, json, sys, torch
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
ROOT = "/data/ndloc_bk/NLHV/ntVan/demo"; os.environ["HF_HOME"] = os.path.join(ROOT, "cache/huggingface")
from unsloth import FastLanguageModel
rows = json.load(open(os.path.join(ROOT, "output/train_v5/llm_data_scene_v5.json"), encoding="utf-8"))[:2]
def run(label, **kw):
    print(f"\n##### {label}", flush=True)
    model, tok = FastLanguageModel.from_pretrained("unsloth/gemma-3-12b-it-unsloth-bnb-4bit", max_seq_length=2048, load_in_4bit=True, cache_dir=os.path.join(ROOT, "cache"), **kw)
    model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16, lora_dropout=0, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], use_gradient_checkpointing="unsloth", random_state=3407)
    texts = [tok.apply_chat_template([{"role":"system","content":r["instruction"]},{"role":"user","content":r["input"]},{"role":"assistant","content":r["output"]}], tokenize=False) for r in rows]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    ids, am = enc.input_ids, enc.attention_mask
    labels = ids.clone(); labels[am == 0] = -100
    for mode in ("eval", "train"):
        getattr(model, mode)()
        with torch.enable_grad():
            out = model(input_ids=ids, attention_mask=am, labels=labels)
            loss = float(out.loss)
            lg = out.logits.detach().float()
            # leakage test: change token at position p+5 of sample 0, compare logits at position p
            p = 200; ids2 = ids.clone(); ids2[0, p+5] = (ids2[0, p+5] + 7) % 200000 + 10
            lg2 = model(input_ids=ids2, attention_mask=am).logits.detach().float()
            diff = float((lg[0, :p+1] - lg2[0, :p+1]).abs().max())
            ce_single = float(torch.nn.functional.cross_entropy(lg[0, :am[0].sum()-1], ids[0, 1:am[0].sum()]))
        print(f"  mode={mode}: loss(batch,padded)={loss:.3f}  CE(sample0,manual)={ce_single:.3f}  max|dlogits| at positions<=p when token p+5 changed = {diff:.4f}  -> {'NON-CAUSAL (LEAK)' if diff > 1e-2 else 'causal ok'}", flush=True)
    del model; torch.cuda.empty_cache()
run("default attn (nhu train_refiner_v5.py)")
run("attn_implementation=eager (nhu finetune_gemma_v6.py)", attn_implementation="eager")
print("PROBE4_DONE", flush=True)
