"""Chi phi ASR do o cung don vi = gold record (data/en-vi-speaker-with-time-pronouns):
 nmt_goldEN = vietsub_raw (mBART tren EN da giong gold) | nmt_asrEN = rough.srt (mBART tren EN ASR, gop theo overlap thoi gian)
 e5_asrEN = v1b_baseline.srt. Chay: van_env/bin/python docs/eval/audit_2026-09-02/asr_cost_record_level.py"""
import json, srt, re, sacrebleu
E5 = ["movie_054", "movie_081", "movie_312", "movie_104", "movie_090", "movie_311", "movie_291", "movie_124", "movie_336", "movie_170"]
def load(p):
    return [(s.start.total_seconds(), s.end.total_seconds(), re.sub(r'\s+', ' ', s.content).strip())
            for s in srt.parse(open(p, encoding='utf-8').read())]
H = {'nmt_goldEN': [], 'nmt_asrEN': [], 'e5_asrEN': []}; R = []
for mv in E5:
    gold = json.load(open(f'data/en-vi-speaker-with-time-pronouns/{mv}.json', encoding='utf-8'))
    ro = load(f'demo/output/eval_e5/{mv}/rough.srt'); e5 = load(f'demo/output/eval_e5/{mv}/v1b_baseline.srt')
    for g in gold:
        gs, ge = float(g['start']), float(g['end'])
        ha = ' '.join(x[2] for x in ro if min(ge, x[1]) - max(gs, x[0]) > 0)
        he = ' '.join(x[2] for x in e5 if min(ge, x[1]) - max(gs, x[0]) > 0)
        if not ha: continue
        R.append(g['vietnamese']); H['nmt_goldEN'].append(g['vietsub_raw'] or ''); H['nmt_asrEN'].append(ha); H['e5_asrEN'].append(he)
for k, h in H.items():
    print(f"{k:12s} n={len(h)} BLEU={sacrebleu.corpus_bleu(h,[R]).score:.2f} chrF={sacrebleu.corpus_chrf(h,[R]).score:.2f}")
