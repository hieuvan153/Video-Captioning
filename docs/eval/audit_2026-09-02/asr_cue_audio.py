"""Dac trung audio cua tung cue GT: do phu Silero-VAD (nguong 0.2 nhu pipeline), RMS dB, spectral flatness,
ti le nang luong hai am (HPSS, proxy nhac nen). So sanh cue dung / mat mot phan / mat hoan toan.
Can <scratch>/asr_cue_errors.json va <scratch>/wav/<movie>.wav. Chay: van_env/bin/python docs/eval/audit_2026-09-02/asr_cue_audio.py <scratch_dir>"""
import json, sys, numpy as np, soundfile as sf, librosa, torch, collections, statistics as st_
S = sys.argv[1]; cues = json.load(open(f'{S}/asr_cue_errors.json'))['cues']
model, utils = torch.hub.load(repo_or_dir='demo/model/ASR/silero-vad', model='silero_vad', onnx=True, source='local'); get_ts = utils[0]
by = collections.defaultdict(list)
for c in cues: by[c[0]].append(c)
rows = []
for mv, cs in sorted(by.items()):
    wav, sr = sf.read(f'{S}/wav/{mv}.wav', dtype='float32')
    if wav.ndim > 1: wav = wav.mean(1)
    if sr != 16000: wav = librosa.resample(wav, orig_sr=sr, target_sr=16000); sr = 16000
    cov = np.zeros(len(wav), dtype=bool)
    for t in get_ts(torch.from_numpy(wav), model, sampling_rate=sr, threshold=0.2): cov[t['start']:t['end']] = True
    for (m, ci, n, dl, sb, s0, e0, raw) in cs:
        a, b = int(s0 * sr), int(min(len(wav), e0 * sr))
        if b - a < sr // 10: continue
        seg = wav[a:b]; H, P = librosa.effects.hpss(seg) if len(seg) > 2048 else (seg, seg * 0)
        rows.append(dict(mv=m, ci=ci, n=n, dl=dl, sb=sb, music='♪' in raw, vad=float(cov[a:b].mean()), rms=float(20 * np.log10(np.sqrt((seg ** 2).mean()) + 1e-9)),
                         flat=float(librosa.feature.spectral_flatness(y=seg).mean()), harm=float((H ** 2).sum() / ((H ** 2).sum() + (P ** 2).sum() + 1e-9)), missed=dl == n))
json.dump(rows, open(f'{S}/asr_cue_audio.json', 'w'))
def summ(name, rs):
    if rs: print(f"{name:34s} n={len(rs):5d} VADcov={st_.mean(r['vad'] for r in rs):.2f} (<0.5: {100*sum(r['vad']<0.5 for r in rs)/len(rs):4.1f}%) rms={st_.mean(r['rms'] for r in rs):6.1f}dB harm={st_.mean(r['harm'] for r in rs):.2f}")
hit = [r for r in rows if r['dl'] == 0 and r['sb'] == 0]; miss = [r for r in rows if r['missed']]
summ('cues fully correct', hit); summ('cues fully missed', miss); summ('  missed & music', [r for r in miss if r['music']]); summ('  missed, not music, <=2 words', [r for r in miss if not r['music'] and r['n'] <= 2])
nm = [r for r in rows if not r['music'] and r['n'] > 2]
for lo, hi in [(0, .5), (.5, .65), (.65, .8), (.8, 1.01)]:
    b = [r for r in nm if lo <= r['harm'] < hi]
    if b: print(f"harmonic ratio [{lo},{hi}): cues={len(b)} del%={100*sum(r['dl'] for r in b)/sum(r['n'] for r in b):.1f}")
