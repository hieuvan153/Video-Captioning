"""Phan tich loi LLM refine tren E5 (khong GPU):
 1) ti le dong doi / chi doi dai tu / doi noi dung; 2) per-gold-record chrF rough vs refined + 14 hoi quy nang nhat;
 3) lech dong trong chunk (output k giong rough k+-1 hon rough k) + chunk sap khoi (>=80% dong giong nhau, guard chua bat);
 4) thu 4 luat "chi nhan sua khi ..." cham bang giao thuc thesis_score (BLEU + F1 dai tu, khong COMET).
Chay: van_env/bin/python docs/eval/audit_2026-09-02/llm_errors_e5.py [--arm v1b_baseline]"""
import json, srt, glob, os, re, sys, collections, argparse, sacrebleu
sys.path.insert(0, 'demo'); os.environ.setdefault('HF_HOME', 'demo/cache/huggingface')
from EVAL.pronoun_lexicon import extract_pronouns
from EVAL import thesis_score as TS
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='v1b_baseline'); ap.add_argument('--root', default='demo/output/eval_e5')
ap.add_argument('--variants_dir', default='/tmp/audit_variants'); a = ap.parse_args()
def load(p):
    return [(s.start.total_seconds(), s.end.total_seconds(), re.sub(r'\s+', ' ', s.content).strip()) for s in srt.parse(open(p, encoding='utf-8').read())]
def strip_pron(t):
    ws = re.sub(r'\W+', ' ', t.lower()).split(); pr = set(w for x in extract_pronouns(t) for w in x.split()); return [w for w in ws if w not in pr]
n_lines = n_chg = pron_only = 0; deltas = []; worst = []
for m in sorted(glob.glob(a.root + '/movie_*')):
    mv = os.path.basename(m); ro = load(f'{m}/rough.srt'); ba = load(f'{m}/{a.arm}.srt')
    gold = json.load(open(f'data/en-vi-speaker-with-time-pronouns/{mv}.json', encoding='utf-8'))
    for r, b in zip(ro, ba):
        n_lines += 1
        if r[2] != b[2]:
            n_chg += 1; pron_only += strip_pron(r[2]) == strip_pron(b[2])
    for g in gold:
        gs, ge = float(g['start']), float(g['end'])
        hr = ' '.join(x[2] for x in ro if min(ge, x[1]) - max(gs, x[0]) > 0); hb = ' '.join(x[2] for x in ba if min(ge, x[1]) - max(gs, x[0]) > 0)
        if not hr: continue
        d = sacrebleu.sentence_chrf(hb, [g['vietnamese']]).score - sacrebleu.sentence_chrf(hr, [g['vietnamese']]).score; deltas.append(d)
        if d < -15: worst.append((d, mv, g['english'][:70], hr[:90], hb[:90], g['vietnamese'][:90]))
print(f"lines {n_lines} changed {n_chg} pronoun-only {pron_only} content-changed {n_chg-pron_only}")
print(f"gold records {len(deltas)} improved(>+1) {sum(d>1 for d in deltas)} regressed(<-1) {sum(d<-1 for d in deltas)} mean {sum(deltas)/len(deltas):.2f} big(<-15) {len(worst)}")
for w in sorted(worst)[:14]: print('  d=%.1f %s\n    EN: %s\n    RO: %s\n    LLM: %s\n    REF: %s' % w)
# 3) shift + collapse from debug json
sh_tot = col_tot = 0
for m in sorted(glob.glob(a.root + '/movie_*')):
    dbg = json.load(open(f'{m}/{a.arm}.srt.json', encoding='utf-8')); sh = col = 0
    for d in dbg:
        if 'translations' not in d: continue
        tr = d['translations']; rough = [t['rough_vietnamese'] for t in tr]; out = [t['refined_vietnamese'] for t in tr]
        for k in range(len(tr)):
            if out[k].strip() == rough[k].strip(): continue
            s_self = sacrebleu.sentence_chrf(out[k], [rough[k]]).score
            s_nb = max((sacrebleu.sentence_chrf(out[k], [rough[j]]).score for j in (k - 1, k + 1) if 0 <= j < len(tr)), default=0)
            sh += s_nb > 60 and s_nb > s_self + 25
        if len(out) >= 5 and collections.Counter(o.strip() for o in out).most_common(1)[0][1] >= 0.8 * len(out) and sum(t['fallback_used'] for t in tr) < len(tr): col += 1
    print(f"{os.path.basename(m)} shifted_lines={sh} collapsed_chunks_missed_by_guard={col}"); sh_tot += sh; col_tot += col
print(f"TOTAL shifted_lines={sh_tot} collapsed_chunks={col_tot}")
# 4) selective acceptance variants
V = {'pron_only': lambda r, b: b if strip_pron(r) == strip_pron(b) else r,
     'chrf60': lambda r, b: b if sacrebleu.sentence_chrf(b, [r]).score >= 60 else r,
     'chrf40': lambda r, b: b if sacrebleu.sentence_chrf(b, [r]).score >= 40 else r,
     'lenok': lambda r, b: b if 0.6 <= len(b.split()) / max(1, len(r.split())) <= 1.6 else r}
for m in sorted(glob.glob(a.root + '/movie_*')):
    mv = os.path.basename(m); ro = list(srt.parse(open(f'{m}/rough.srt', encoding='utf-8').read())); ba = list(srt.parse(open(f'{m}/{a.arm}.srt', encoding='utf-8').read()))
    os.makedirs(f'{a.variants_dir}/{mv}', exist_ok=True)
    for name, fn in V.items():
        out = [srt.Subtitle(index=b.index, start=b.start, end=b.end, content=fn(re.sub(r'\s+', ' ', r.content).strip(), re.sub(r'\s+', ' ', b.content).strip())) for r, b in zip(ro, ba)]
        open(f'{a.variants_dir}/{mv}/{name}.srt', 'w', encoding='utf-8').write(srt.compose(out))
for arm in ['rough.srt', a.arm + '.srt'] + list(V):
    hyps = []; refs = []; tp = fp = fn = 0
    for m in sorted(glob.glob(a.root + '/movie_*')):
        mv = os.path.basename(m); f = arm if arm.endswith('.srt') else f'{a.variants_dir}/{mv}/{arm}.srt'
        for s, h, r in TS.scene_texts(m, mv, f):
            hyps.append(h); refs.append(r); t, p_, n_ = TS.pronoun_sets_scene(h, r); tp += t; fp += p_; fn += n_
    P = tp / (tp + fp); Rc = tp / (tp + fn)
    print(f"{arm:18s} BLEU={sacrebleu.corpus_bleu(hyps,[refs]).score:.2f} chrF={sacrebleu.corpus_chrf(hyps,[refs]).score:.2f} F1={2*P*Rc/(P+Rc):.4f}")
