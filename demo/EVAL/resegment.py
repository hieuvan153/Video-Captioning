"""Cat lai cue tu moc tu da luu (words_json) -> SRT. Khong can GPU, khong doi chu, chi doi ranh gioi.
CLI: resegment.py words.json out.srt <gap_s|-> <max_s|->"""
import datetime as dt, io, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ASR"))
import srt
from asr_fw_infer import split_words

f = lambda x: None if x == "-" else float(x)
ws = json.load(io.open(sys.argv[1], encoding="utf-8"))
subs = []
for g in split_words(ws, f(sys.argv[3]), f(sys.argv[4])):
    t = "".join(w["w"] for w in g).strip()
    if t:
        subs.append(srt.Subtitle(index=len(subs) + 1, content=t,
                                 start=dt.timedelta(seconds=g[0]["s"]),
                                 end=dt.timedelta(seconds=max(g[-1]["e"], g[0]["s"] + 0.01))))
io.open(sys.argv[2], "w", encoding="utf-8").write(srt.compose(subs))
print(f"{sys.argv[2].rsplit('/',1)[-1]}: {len(subs)} cue (gap={sys.argv[3]} max={sys.argv[4]})")
