# Su co train v5: loss sup ve 0, adapter chi hoc CHEP (2026-08-19)

## Trieu chung
Train QLoRA gemma-3-12b-it tren data v5 (11792 mau sach, sub-chunk <=24):
loss 6.29 (step 20) -> 0.0656 (step 140) -> **0.0000 tu step 540** (67/110
diem log). Chay het 10h40, 4 checkpoint.

## Bang chung adapter hong
- checkpoint-2211 (3 epoch) sinh tren movie_001: chi sua **4.2%** dong,
  trong khi gold khac rough **91%** dong.
- Sinh tren chinh MAU TRAIN: 3/3 mau ra dung ban rough (`SINH == ROUGH: True`,
  `SINH == GOLD: False`). Day la bang chung quyet dinh.
- lora_B norm: epoch1 0.1399 -> epoch4 0.1470 (chi +5%) => gradient tat sau
  epoch 1.

## Da loai tru (do dac, khong doan)
| Nghi van | Ket qua |
|---|---|
| Mask hong toan phan | Loai: loss co giam that 6.29->0.065 truoc khi ve 0 |
| Truncation 2048 | Loai: p50 718 token, max 1504, 0% vuot nguong |
| Task von la copy | Loai: chi 7.5% token trung vi tri voi rough |
| Prompt train != inference | Loai: render ra 2 chuoi giong het tung byte |
| Ham mask unsloth sai | Loai: goi doc lap ra dung 23 token = gold output |
| TRL dung collator VLM ghi de nhan | Loai: `_is_vision_dataset`=False (dataset
  chi co truong "text") -> dung DataCollatorForLanguageModeling |
| Nhan la rough thay vi gold | Loai: decode nhan tu batch THAT ra gold cua row 43 |

Do loss 3 mo hinh tren cung data train (32 mau):
base (LoRA chua hoc) 6.2154 | checkpoint-2948 **0.0000** | adapter goc ntVan
5.9726. Adapter goc gan bang base la VO LY => duong tinh loss nay cho so rac;
chi co bang chung GENERATION la dang tin.

## Nguyen nhan goc
Ca 3 script finetune cua tac gia (ALMA/finetune_gemma_v6.py,
finetune_gemma.py, finetune_gemma4_26b_text_v6.py) deu **KHONG** dung
`train_on_responses_only` (grep = 0) — ho train tren toan bo text. Ta tu them
ham nay vao vi no la "thuc hanh tot" thong thuong; tren dung ngan xep nay
(Gemma-3 wrapper vision-language + TRL 0.24 + unsloth) no tao ra ham muc tieu
thoai hoa: loss sup ve 0 trong ~300 step, model chi hoc chep lai rough.

## Sua
1. Bo `train_on_responses_only` khoi demo/LLM/train_refiner_v5.py, bam dung
   cong thuc tac gia (train toan bo text, 8 epoch, lr 5e-5, inverse_sqrt,
   warmup_ratio 0.1, wd 0.01, seed 3407, batch 4x4).
2. Them `--max_steps` de SMOKE TEST truoc khi chay dai.
3. Checkpoint hong doi ten -> demo/model/refiner_v5_BROKEN_copymodel (giu lai).

## Bai hoc
- LUON smoke test ~60 step va kiem loss truoc khi cam ket chay 11 gio.
- Tieu chi nghiem thu: loss giam dan nhung GIU TREN ~0.5, khong sup ve 0.
- Khong "cai tien" cong thuc da duoc chung minh chay duoc neu chua co bang
  chung; moi bien the phai qua smoke test rieng.
- Loss la proxy; GENERATION moi la su that. Khi hai cai mau thuan, tin
  generation.

## Smoke 60 step sau sua eager (2026-09-02, Task 0 E0)

Lenh: docs/eval/audit_2026-09-02/smoke_train.sh (train_refiner_v5.py + attn_implementation="eager", du lieu train_v5, batch 4 x accum 4).

| step | loss | grad_norm |
|---|---|---|
| 10 | 2.8948 | 1.49 |
| 20 | 2.0106 | 0.69 |
| 30 | 1.7730 | 0.47 |
| 40 | 1.6089 | 0.42 |
| 50 | 1.4737 | 0.47 |
| 60 | 1.3797 | 0.39 |

Gate (step 10 trong [1.8, 4.0]; step 60 > 1.0; khong moc < 0.3): **DAT**. Truoc khi sua, cung script sup 8 -> 0 trong 60 step. Ket luan: thieu eager la nguyen nhan; chay full adapter sach bang demo/LLM/run_train_clean.sh.
