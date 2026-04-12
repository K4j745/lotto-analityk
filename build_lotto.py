import json, itertools, random, warnings, os, requests
warnings.filterwarnings('ignore')
from collections import Counter, defaultdict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

print("Pobieram dane z mbnet.com.pl...")
r = requests.get('http://www.mbnet.com.pl/dl.txt', timeout=30)
draws = []
for line in r.text.strip().split('\n'):
    parts = line.strip().split(' ', 2)
    if len(parts) == 3:
        nums = sorted(map(int, parts[2].split(',')))
        draws.append({"id": int(parts[0].rstrip('.')), "date": parts[1], "numbers": nums})
total = len(draws)
print(f"Zaladowano {total} losowan. Ostatnie: {draws[-1]['date']} -> {draws[-1]['numbers']}")

freq = Counter()
for d in draws: freq.update(d['numbers'])
freq_list = [[i, freq.get(i,0)] for i in range(1,50)]
recent = draws[-365:]
rfq = Counter()
for d in recent: rfq.update(d['numbers'])
recent_freq = [[i, rfq.get(i,0)] for i in range(1,50)]
decade_freq = {}
for d in draws:
    year = int(d['date'].split('.')[-1]); dec = (year // 10) * 10
    if dec not in decade_freq: decade_freq[dec] = Counter()
    decade_freq[dec].update(d['numbers'])
decade_data = {str(dec): {str(i): cnt.get(i,0) for i in range(1,50)} for dec, cnt in decade_freq.items()}
pair_count = Counter()
for d in draws:
    for pair in itertools.combinations(d['numbers'], 2): pair_count[pair] += 1
top_pairs = [[list(p), cnt] for p, cnt in pair_count.most_common(20)]
triple_count = Counter()
for d in draws:
    for triple in itertools.combinations(d['numbers'], 3): triple_count[triple] += 1
top_triples = [[list(t), cnt] for t, cnt in triple_count.most_common(10)]
markov = defaultdict(Counter)
for i in range(len(draws) - 1):
    for c in draws[i]['numbers']:
        for n in draws[i+1]['numbers']: markov[c][n] += 1
markov_top = {str(num): [[n, cnt] for n, cnt in markov[num].most_common(3)] for num in range(1,50)}
random.seed(42)
boot_ci = {}
for num in range(1,50):
    presence = [1 if num in d['numbers'] else 0 for d in draws]; n = len(presence)
    boot_means = sorted([sum(random.choices(presence, k=n))/n for _ in range(500)]); mean = sum(presence)/n
    boot_ci[str(num)] = [round(boot_means[12]*100,2), round(mean*100,2), round(boot_means[487]*100,2)]
cooc = np.zeros((49,49))
for d in draws:
    for a in d['numbers']:
        for b in d['numbers']:
            if a != b: cooc[a-1][b-1] += 1
labels = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(StandardScaler().fit_transform(cooc))
clusters = {str(k): [] for k in range(5)}
for num, label in enumerate(labels): clusters[str(label)].append(num+1)
WINDOW = 6
def mf(draw_idx):
    feats = []
    for w in range(WINDOW, 0, -1):
        idx = draw_idx - w
        feats.extend([1 if n in draws[idx]['numbers'] else 0 for n in range(1,50)] if idx >= 0 else [0]*49)
    roll = Counter()
    for d in draws[max(0, draw_idx-30):draw_idx]: roll.update(d['numbers'])
    feats.extend([roll.get(n,0) for n in range(1,50)]); return feats
print("Trenuję Random Forest (chwila)...")
X = np.array([mf(i) for i in range(WINDOW, total-1)], dtype=np.float32)
Y = np.array([[1 if n in draws[i+1]['numbers'] else 0 for n in range(1,50)] for i in range(WINDOW, total-1)], dtype=np.float32)
split = len(X) - 300; X_tr, X_te, Y_tr, Y_te = X[:split], X[split:], Y[:split], Y[split:]
Xn = np.array([mf(total-1)], dtype=np.float32)
rf_probs = np.zeros(49); rf_accs = []
for i in range(49):
    y = Y_tr[:, i]
    if y.sum() < 5: rf_probs[i] = float(y.mean()); continue
    clf = RandomForestClassifier(n_estimators=40, max_depth=5, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y); rf_probs[i] = clf.predict_proba(Xn)[0][1]
    rf_accs.append(float((clf.predict(X_te) == Y_te[:,i]).mean()))
rf_top6 = sorted([r+1 for r in sorted(range(49), key=lambda i: rf_probs[i], reverse=True)[:6]])
rf_probs_out = {str(i+1): round(float(rf_probs[i])*100, 2) for i in range(49)}
avg_acc = round(sum(rf_accs)/len(rf_accs)*100, 2)
print(f"RF Top6={rf_top6}, avg_acc={avg_acc}%")

J = lambda x: json.dumps(x, separators=(',',':'))

with open('template.html', 'r', encoding='utf-8') as f:
    result = f.read()

draws_rows = ""
for d in reversed(draws[-20:]):
    balls_html = "".join('<div class="dball">' + str(n) + '</div>' for n in d["numbers"])
    draws_rows += (
        '<tr><td style="color:var(--txm);font-variant-numeric:tabular-nums">' + str(d["id"]) + '</td>'
        '<td style="white-space:nowrap">' + d["date"] + '</td>'
        '<td><div class="dnums">' + balls_html + '</div></td></tr>'
    )

result = result.replace('{{FREQ}}', J(freq_list))
result = result.replace('{{REC}}', J(recent_freq))
result = result.replace('{{DEC}}', J(decade_data))
result = result.replace('{{LAST20}}', J(draws[-20:]))
result = result.replace('{{PAIRS}}', J(top_pairs))
result = result.replace('{{TRIPLES}}', J(top_triples))
result = result.replace('{{MARKOV}}', J(markov_top))
result = result.replace('{{CLUSTERS}}', J(clusters))
result = result.replace('{{BOOT}}', J(boot_ci))
result = result.replace('{{RF}}', J(rf_probs_out))
result = result.replace('{{RF6}}', J(rf_top6))
result = result.replace('{{TOTAL}}', str(total))
result = result.replace('{{SPLIT}}', str(split))
result = result.replace('{{RFACC}}', str(avg_acc))
result = result.replace('{{RF6STR}}', str(rf_top6))
result = result.replace('{{LAST_DATE}}', draws[-1]['date'])
result = result.replace('{{DRAWS_ROWS}}', draws_rows)

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(result)
print(f"Gotowe! index.html ({len(result)} bajtow)")
