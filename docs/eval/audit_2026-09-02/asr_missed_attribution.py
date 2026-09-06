"""Quy loi ASR ve tung nguyen nhan cho cac cue GT bi mat HOAN TOAN, tren 10 tap E5.
Can: asr_cue_errors.json + asr_cue_audio.json (tu asr_wer_e5 phan tich cue, VAD/spectral) va thu muc rerun ASR
(asr054/, asr_movie_XXX/) chua segment_info.json (moi segment Whisper TRUOC bo loc) + SRT cuoi.
Chay: van_env/bin/python docs/eval/audit_2026-09-02/asr_missed_attribution.py <scratch_dir>"""
import json, re, srt, os, sys, collections
S = sys.argv[1]
def norm(t): return ' '.join(re.sub(r"[^a-z0-9' ]+", ' ', re.sub(r'\([^)]*\)|\[[^\]]*\]|<[^>]+>|\b[A-Z][A-Z .\']{1,30}:', '', t).replace('♪', '').lower()).split())
d = json.load(open(f'{S}/asr_cue_errors.json')); audio = json.load(open(f'{S}/asr_cue_audio.json'))
vad = {(r['mv'], r['ci']): r for r in audio}
tot = collections.Counter(); per = {}
for mv in sorted(set(m[0] for m in d['missed'])):
    dirn = f'{S}/asr054' if mv == 'movie_054' else f'{S}/asr_{mv}'
    if not os.path.exists(f'{dirn}/segment_info.json'): print(mv, 'chua co rerun'); continue
    seg = json.load(open(f'{dirn}/segment_info.json'))
    bad = lambda s: s['avg_logprob'] < -1.0 or s['no_speech_prob'] > 0.9 or s['compression_ratio'] > 6
    kt = ' '.join(norm(s['text']) for s in seg if not bad(s)); dt = ' '.join(norm(s['text']) for s in seg if bad(s))
    new = ' '.join(norm(s.content) for s in srt.parse(open(f'{dirn}/{mv}.(Tiếng Anh).srt', encoding='utf-8').read()))
    c = collections.Counter()
    for m in [m for m in d['missed'] if m[0] == mv]:
        ws = norm(m[5]).split(); key = ' '.join(ws[:3]) if len(ws) >= 2 else ' '.join(ws); r = vad.get((mv, m[1]))
        if '♪' in m[5]: c['loi bai hat ♪'] += 1
        elif not key: c['rong sau chuan hoa'] += 1
        elif key in new: c['rerun hom nay nhan ra (khac run cu)'] += 1
        elif key in kt: c['Whisper sinh ra, bi garbage-list xoa'] += 1
        elif key in dt: c['Whisper sinh ra, bi loc logprob/no_speech/compression'] += 1
        elif r and r['vad'] < 0.5: c['VAD bo (ngoai vung speech)'] += 1
        else: c['trong vung VAD nhung Whisper khong sinh'] += 1
    per[mv] = dict(c); tot.update(c)
    print(mv, dict(c))
print('\nTONG', dict(tot), 'tong cue mat hoan toan', sum(tot.values()))
