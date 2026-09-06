"""Thong ke chat luong gold: dataset en-vi-speaker-with-time-pronouns (356 phim) + do phu so voi phu de VI chinh thuc tren 10 tap E5.
Chay: van_env/bin/python docs/eval/audit_2026-09-02/gold_quality.py"""
import json, glob, re, collections, os, statistics, srt, unicodedata
D = 'data/en-vi-speaker-with-time-pronouns'; n = age_null = overlap = ratio_out = spk = paren = con_cls = lab = lab_missing = 0; dur = []
for f in sorted(glob.glob(D + '/*.json')):
    prev_end = -1
    for r in json.load(open(f, encoding='utf-8')):
        n += 1; vi = (r.get('vietnamese') or '').strip(); en = (r.get('english') or '').strip()
        age_null += r.get('age') is None; s, e = float(r['start']), float(r['end']); dur.append(e - s); overlap += s < prev_end; prev_end = max(prev_end, e)
        rt = len(vi.split()) / max(1, len(en.split())); ratio_out += rt > 3 or rt < 0.33
        spk += bool(re.search(r'\b[A-Z][A-Z .]+:', en)); paren += '(' in en
        labels = [t.strip().lower() for fld in ('pronouns_subject', 'pronouns_object') for t in str(r.get(fld) or '').split(',') if t.strip()]
        lab += len(labels); lab_missing += sum(not re.search(r'(?<!\w)' + re.escape(t) + r'(?!\w)', vi.lower()) for t in labels)
        con_cls += 'con' in labels and bool(re.search(r'\bcon (?:mồi|chó|mèo|gà|bò|ngựa|cá|chim|vật|số|đường|người)\b', vi.lower()))
print(f"records {n}; age/gender null {age_null} ({100*age_null/n:.1f}%); time-overlap {overlap}; dur median {statistics.median(dur):.2f}s p90 {sorted(dur)[int(.9*len(dur))]:.2f}s")
print(f"len-ratio outliers {ratio_out}; EN with NAME: {spk}; EN with (..) {paren}; pronoun labels {lab}, not in text {lab_missing}; 'con' labelled but classifier (narrow regex) {con_cls}")
E5 = {"movie_054": "S01E017_Nhu_Thuật_bọc_bong_bóng_và_chăm_lo", "movie_081": "S02E021_Trái_tim_tan_vỡ_và_quái_vật_đất_nung", "movie_312": "S03E001_Đồn_trưởng_mới", "movie_104": "S03E002_Tủ_đựng_chổi_và_cờ_tỷ_phú_của_Satan", "movie_090": "S03E011_Gà_sống_gà_rán_và_thánh_hôn", "movie_311": "S03E015_Đồn_98", "movie_291": "S03E017_Adrian_Pimento", "movie_124": "S04E014_Đậu_má_và_sự_chấp_thuận_vô_điều_kiện_của_cơ_quan_c", "movie_336": "S05E015_Bá_chủ_ô_chữ", "movie_170": "S07E002_Bàn_quay_roulette_và_chó_biết_chơi_piano"}
subs = {unicodedata.normalize('NFC', os.path.basename(p)): p for p in glob.glob('data/Movie/sub/*.srt')}; tw_off = tw_gold = 0
for mv, ep in E5.items():
    off = list(srt.parse(open(subs[unicodedata.normalize('NFC', ep + '.Tiếng_Việt.srt')], encoding='utf-8').read()))
    gold = json.load(open(f'{D}/{mv}.json', encoding='utf-8')); wo = sum(len(re.sub(r'<[^>]+>', '', s.content).split()) for s in off); wg = sum(len(r['vietnamese'].split()) for r in gold)
    tw_off += wo; tw_gold += wg; print(f"{mv} official cues={len(off)} words={wo} | gold recs={len(gold)} words={wg} ({100*wg/wo:.0f}%) merged={sum('  ' in r['vietnamese'] for r in gold)}")
print(f"TOTAL gold/official words {100*tw_gold/tw_off:.1f}%")
