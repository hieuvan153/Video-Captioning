"""Probe 3b: CE cua base model khi chen them mot <bos> o dau (nhu batch cua SFTTrainer trl 0.24: input bat dau [2, 2, ...])."""
import os, json, torch
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
ROOT = "/data/ndloc_bk/NLHV/ntVan/demo"; os.environ["HF_HOME"] = os.path.join(ROOT, "cache/huggingface")
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained("unsloth/gemma-3-12b-it-unsloth-bnb-4bit", max_seq_length=2048, load_in_4bit=True, cache_dir=os.path.join(ROOT, "cache"))
FastLanguageModel.for_inference(model)
rows = json.load(open(os.path.join(ROOT, "output/train_v5/llm_data_scene_v5.json"), encoding="utf-8"))[:4]
bos = tok.bos_token_id
for r in rows:
    text = tok.apply_chat_template([{"role":"system","content":r["instruction"]},{"role":"user","content":r["input"]},{"role":"assistant","content":r["output"]}], tokenize=False)
    ids1 = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    ids2 = torch.cat([torch.tensor([[bos]], device="cuda"), ids1], dim=1)
    res = []
    for ids in (ids1, ids2):
        with torch.no_grad(): lg = model(input_ids=ids).logits.float()
        res.append(float(torch.nn.functional.cross_entropy(lg[0, :-1], ids[0, 1:])))
    print(f"single_bos {ids1[0,:3].tolist()} CE={res[0]:.3f} | double_bos {ids2[0,:3].tolist()} CE={res[1]:.3f}", flush=True)
print("PROBE3_DONE", flush=True)
