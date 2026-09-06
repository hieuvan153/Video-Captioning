"""Probe 2: so loss ma SFTTrainer (trl 0.24 + unsloth 2026.5.2 + transformers 4.57.6) bao cao
voi CE tinh tay tren CUNG mot batch, va xem batch['labels'] co bi mask gan het khong."""
import os, json, torch
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
ROOT = "/data/ndloc_bk/NLHV/ntVan/demo"; os.environ["HF_HOME"] = os.path.join(ROOT, "cache/huggingface")
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
model, tok = FastLanguageModel.from_pretrained("unsloth/gemma-3-12b-it-unsloth-bnb-4bit", max_seq_length=2048, load_in_4bit=True, cache_dir=os.path.join(ROOT, "cache"))
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16, lora_dropout=0.1, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], use_gradient_checkpointing="unsloth", random_state=3407)
rows = json.load(open(os.path.join(ROOT, "output/train_v5/llm_data_scene_v5.json"), encoding="utf-8"))[:16]
def to_text(r):
    return tok.apply_chat_template([{"role":"system","content":r["instruction"]},{"role":"user","content":r["input"]},{"role":"assistant","content":r["output"]}], tokenize=False)
ds = Dataset.from_list([{"text": to_text(r)} for r in rows])
trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=ds, args=SFTConfig(dataset_text_field="text", max_length=2048, per_device_train_batch_size=4, gradient_accumulation_steps=4, max_steps=1, output_dir="/tmp/probe_out", bf16=True, optim="adamw_8bit", report_to="none", logging_steps=1))
dl = trainer.get_train_dataloader(); batch = next(iter(dl))
print("batch keys", list(batch.keys()), {k: tuple(v.shape) for k, v in batch.items() if hasattr(v, 'shape')}, flush=True)
lab = batch["labels"]; print("labels: total", lab.numel(), "masked(-100)", int((lab == -100).sum()), "pad_id", tok.pad_token_id, "n==pad", int((lab == tok.pad_token_id).sum()), flush=True)
print("first 12 labels", lab[0, :12].tolist(), "input", batch["input_ids"][0, :12].tolist(), flush=True)
print("labels equal input_ids where not masked:", bool(((lab == batch["input_ids"]) | (lab == -100)).all()), flush=True)
model.eval(); b = {k: v.cuda() for k, v in batch.items()}
with torch.no_grad():
    loss_trainer = trainer.compute_loss(model, dict(b))
    print("trainer.compute_loss =", float(loss_trainer if not isinstance(loss_trainer, tuple) else loss_trainer[0]), flush=True)
    out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"])
    print("model(labels=).loss =", float(out.loss), flush=True)
    logits = out.logits.float(); l = b["labels"]
    ce = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), l[:, 1:].reshape(-1), ignore_index=-100)
    print("manual shifted CE =", float(ce), flush=True)
    nib = int((l[:, 1:] != -100).sum()); print("num_items_in_batch(manual) =", nib, flush=True)
    try:
        out2 = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"], num_items_in_batch=nib)
        print("model(labels=, num_items_in_batch=) .loss =", float(out2.loss), flush=True)
    except Exception as e: print("num_items_in_batch call failed:", repr(e)[:200], flush=True)
print("PROBE2_DONE", flush=True)
