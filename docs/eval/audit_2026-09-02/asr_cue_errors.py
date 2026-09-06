"""Quy loi ASR ve tung cue GT (global word alignment jiwer -> map tu ref ve cue), phan loai theo dac trung cue
(♪, tag am thanh, <=2 tu, toc do noi, dau/cuoi tap, chong cue truoc) + top tu bi xoa / cap thay the.
Ghi <scratch>/asr_cue_errors.json cho asr_cue_audio.py va asr_missed_attribution.py.
Chay: van_env/bin/python docs/eval/audit_2026-09-02/asr_cue_errors.py <scratch_dir>"""
import srt, glob, os, re, sys, unicodedata, jiwer, collections, json
S = sys.argv[1]; GT = 'ASR/ground_truth_asr'
gts = {unicodedata.normalize('NFC', os.path.basename(p)): p for p in glob.glob(GT + '/*.srt')}
E5 = {"S01E017": "movie_054", "S02E021": "movie_081", "S03E001": "movie_312", "S03E002": "movie_104", "S03E011": "movie_090", "S03E015": "movie_311", "S03E017": "movie_291", "S04E014": "movie_124", "S05E015": "movie_336", "S07E002": "movie_170"}
def norm(t):
    t = re.sub(r'<[^>]+>', '', t); t = re.sub(r'\([^)]*\)', '', t); t = re.sub(r'\[[^\]]*\]', '', t)
    t = re.sub(r'\b[A-Z][A-Z .\']{1,30}:', '', t); t = t.replace('♪', '').lower()
    t = re.sub(r"[^a-z0-9' ]+", ' ', t); return re.sub(r'\s+', ' ', t).strip()
cats = collections.defaultdict(lambda: [0, 0, 0]); subs_pairs = collections.Counter(); del_words = collections.Counter(); missed = []; allcues = []
for code, mv in E5.items():
    cues = list(srt.parse(open(gts[[k for k in gts if k.startswith(code)][0]], encoding='utf-8').read()))
    hyp_words = ' '.join(norm(s.content) for s in srt.parse(open(f'demo/output/eval_e5/{mv}/en.srt', encoding='utf-8').read())).split()
    ref_words = []; cue_of = []
    for ci, c in enumerate(cues):
        ws = norm(c.content).split(); ref_words += ws; cue_of += [ci] * len(ws)
    o = jiwer.process_words(' '.join(ref_words), ' '.join(hyp_words)); per = collections.defaultdict(lambda: [0, 0, 0])
    for al in o.alignments[0]:
        for i in range(al.ref_start_idx, al.ref_end_idx):
            ci = cue_of[i]; per[ci][0] += 1
            if al.type == 'delete': per[ci][1] += 1; del_words[ref_words[i]] += 1
            elif al.type == 'substitute':
                per[ci][2] += 1; j = al.hyp_start_idx + (i - al.ref_start_idx)
                if j < al.hyp_end_idx: subs_pairs[(ref_words[i], hyp_words[j])] += 1
    T = cues[-1].end.total_seconds()
    for ci, c in enumerate(cues):
        n, d, s = per.get(ci, [0, 0, 0])
        if n == 0: continue
        st, en = c.start.total_seconds(), c.end.total_seconds(); raw = c.content
        feats = {'music♪': '♪' in raw, 'sound_tag(..)': bool(re.search(r'\([^)]*\)', raw)), 'short<=2w': n <= 2, 'fast>4w/s': n / max(0.1, en - st) > 4,
                 'first60s': st < 60, 'last60s': en > T - 60, 'overlap_prev_cue': ci > 0 and cues[ci - 1].end.total_seconds() > st + 0.05, 'dash_two_speakers': '\n-' in raw or raw.startswith('-')}
        for k, v in [('ALL', True)] + list(feats.items()):
            if v: cats[k][0] += n; cats[k][1] += d; cats[k][2] += s
        allcues.append((mv, ci, n, d, s, st, en, raw))
        if d == n: missed.append((mv, ci, n, st, en, raw.replace('\n', ' | ')))
print(f"{'category':20s} {'refw':>6s} {'del%':>6s} {'sub%':>6s} {'share_del':>9s}")
for k, (n, d, s) in sorted(cats.items(), key=lambda kv: -kv[1][1]): print(f"{k:20s} {n:6d} {100*d/n:6.1f} {100*s/n:6.1f} {100*d/cats['ALL'][1]:8.1f}%")
print('fully-missed cues', len(missed), 'words', sum(m[2] for m in missed), 'of deletions', cats['ALL'][1])
print('top deleted', del_words.most_common(20)); print('top substitutions', subs_pairs.most_common(20))
json.dump({'missed': missed, 'cues': allcues}, open(f'{S}/asr_cue_errors.json', 'w'), ensure_ascii=False)
