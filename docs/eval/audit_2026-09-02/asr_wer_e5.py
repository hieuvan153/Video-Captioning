"""WER cua ASR (en.srt trong eval_e5) so voi phu de EN chinh thuc (ASR/ground_truth_asr), 10 tap E5.
Chuan hoa: bo nhan NGUOI NOI:, (am thanh), [..], lowercase, bo dau cau. In them S/D/I va ti le dong phan manh.
Chay: van_env/bin/python docs/eval/audit_2026-09-02/asr_wer_e5.py   (tu thu muc ntVan)"""
import os
import srt, glob, os, re, unicodedata, jiwer
GT = 'ASR/ground_truth_asr'
E5 = {"S01E017": "movie_054", "S02E021": "movie_081", "S03E001": "movie_312", "S03E002": "movie_104",
      "S03E011": "movie_090", "S03E015": "movie_311", "S03E017": "movie_291", "S04E014": "movie_124",
      "S05E015": "movie_336", "S07E002": "movie_170"}
gts = {unicodedata.normalize('NFC', os.path.basename(p)): p for p in glob.glob(GT + '/*.srt')}

def norm(t):
    t = re.sub(r'<[^>]+>', '', t); t = re.sub(r'\([^)]*\)', '', t); t = re.sub(r'\[[^\]]*\]', '', t)
    t = re.sub(r'\b[A-Z][A-Z .\']{1,30}:', '', t); t = t.replace('♪', '').lower()
    t = re.sub(r"[^a-z0-9' ]+", ' ', t); return re.sub(r'\s+', ' ', t).strip()

S = D = I = H = 0; frag_end = frag_low = n = 0
for code, mv in E5.items():
    name = [k for k in gts if k.startswith(code)][0]
    ref = ' '.join(norm(s.content) for s in srt.parse(open(gts[name], encoding='utf-8').read()))
    hyp_subs = list(srt.parse(open(f'{os.environ.get("EVAL_DIR","demo/output/eval_e5")}/{mv}/en.srt', encoding='utf-8').read()))
    hyp = ' '.join(norm(s.content) for s in hyp_subs)
    for s in hyp_subs:
        t = re.sub(r'\s+', ' ', s.content).strip(); n += 1
        frag_end += not re.search(r'[.!?…"\')]$', t); frag_low += bool(t) and t[0].islower()
    o = jiwer.process_words(re.sub(r'\s+', ' ', ref).strip(), re.sub(r'\s+', ' ', hyp).strip())
    S += o.substitutions; D += o.deletions; I += o.insertions; H += o.hits
    print(f"{mv} {code} WER={o.wer*100:5.1f}% sub={o.substitutions} del={o.deletions} ins={o.insertions}")
N = S + D + H
print(f"ALL WER={(S+D+I)/N*100:.1f}%  sub={100*S/N:.1f}% del={100*D/N:.1f}% ins={100*I/N:.1f}%")
print(f"ASR lines {n}: no sentence-final punctuation {frag_end} ({100*frag_end/n:.0f}%), lowercase start {frag_low} ({100*frag_low/n:.0f}%)")
