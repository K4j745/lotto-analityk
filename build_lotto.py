import json, itertools, random, warnings, os, requests, sys
warnings.filterwarnings('ignore')
from collections import Counter, defaultdict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ── 1. Pobieranie danych ────────────────────────────────────────────────────
print("Pobieram dane z mbnet.com.pl...")
try:
    r = requests.get('http://www.mbnet.com.pl/dl.txt', timeout=30)
    r.raise_for_status()
except requests.exceptions.Timeout:
    print("BLAD: timeout (>30s)"); sys.exit(1)
except requests.exceptions.ConnectionError as e:
    print(f"BLAD polaczenia: {e}"); sys.exit(1)
except requests.exceptions.HTTPError as e:
    print(f"BLAD HTTP: {e}"); sys.exit(1)
except Exception as e:
    print(f"BLAD: {e}"); sys.exit(1)

draws = []
for line in r.text.strip().split('\n'):
    parts = line.strip().split(' ', 2)
    if len(parts) == 3:
        try:
            nums = sorted(map(int, parts[2].split(',')))
            if len(nums) == 6:
                draws.append({"id": int(parts[0].rstrip('.')), "date": parts[1], "numbers": nums})
        except (ValueError, IndexError):
            continue

total = len(draws)
if total < 100:
    print(f"BLAD: za malo losowan ({total})"); sys.exit(1)
print(f"Zaladowano {total} losowan. Ostatnie: {draws[-1]['date']} -> {draws[-1]['numbers']}")

# ── 2. Czestotliwosci ───────────────────────────────────────────────────────
freq = Counter()
for d in draws:
    freq.update(d['numbers'])

freq_list   = [[i, freq.get(i, 0)] for i in range(1, 50)]

recent = draws[-365:]
rfq = Counter()
for d in recent:
    rfq.update(d['numbers'])
recent_freq = [[i, rfq.get(i, 0)] for i in range(1, 50)]

decade_freq = {}
for d in draws:
    year = int(d['date'].split('.')[-1])
    dec  = (year // 10) * 10
    if dec not in decade_freq:
        decade_freq[dec] = Counter()
    decade_freq[dec].update(d['numbers'])
decade_data = {
    str(dec): {str(i): cnt.get(i, 0) for i in range(1, 50)}
    for dec, cnt in decade_freq.items()
}

last20_data = [{"id": d["id"], "date": d["date"], "numbers": d["numbers"]}
               for d in draws[-20:]]

# ── 3. Tabela HTML (ostatnie 20 wierszy) ────────────────────────────────────
draws_rows = ""
for d in reversed(draws[-20:]):
    nums_html = " ".join(f'<span class="dball">{n}</span>' for n in d['numbers'])
    draws_rows += (f'<tr><td>{d["id"]}</td>'
                   f'<td>{d["date"]}</td>'
                   f'<td><div class="dnums">{nums_html}</div></td></tr>\n')

# ── 4. Association Rules ─────────────────────────────────────────────────────
# Format: [[[n1,n2], count], ...]
pair_count   = Counter()
triple_count = Counter()
for d in draws:
    for pair   in itertools.combinations(d['numbers'], 2):
        pair_count[pair] += 1
    for triple in itertools.combinations(d['numbers'], 3):
        triple_count[triple] += 1

top_pairs   = [[list(p), c] for p, c in pair_count.most_common(20)]
top_triples = [[list(t), c] for t, c in triple_count.most_common(10)]

# ── 5. Lancuch Markowa ──────────────────────────────────────────────────────
# Format: {"num_str": [[next_num, count], ...]}
markov = defaultdict(Counter)
for d in draws:
    nums = d['numbers']
    for a in nums:
        for b in nums:
            if a != b:
                markov[a][b] += 1

markov_data = {
    str(n): [[int(k), int(v)] for k, v in markov[n].most_common(5)]
    for n in range(1, 50)
}

# ── 6. Macierz binarna losowan (total x 49) ─────────────────────────────────
X = np.zeros((total, 49), dtype=np.float32)
for i, d in enumerate(draws):
    for n in d['numbers']:
        X[i, n - 1] = 1.0

# ── 7. KMeans — klastrowanie LICZB 1-49 ─────────────────────────────────────
# Wektor kazde liczby = jej profil wspolwystepowania z innymi liczbami
# Format: {"cluster_id": [num1, num2, ...]} — flat lista liczb per klaster
co_matrix = X.T @ X      # (49 x 49)
np.fill_diagonal(co_matrix, 0)
co_norm   = co_matrix / (total + 1e-9)

km         = KMeans(n_clusters=5, random_state=42, n_init=10)
num_labels = km.fit_predict(co_norm)

clusters_json = defaultdict(list)
for num_idx, label in enumerate(num_labels):
    clusters_json[str(int(label))].append(num_idx + 1)
clusters_json = dict(clusters_json)

# ── 8. Bootstrap CI 95% per liczba ──────────────────────────────────────────
# Format: {"num_str": [ci_low_pct, mean_pct, ci_high_pct]}
rng    = np.random.default_rng(42)
N_BOOT = 1000
boot_data = {}

for num_idx in range(49):
    col = X[:, num_idx]
    boot_means = []
    for _ in range(N_BOOT):
        sample = rng.choice(col, size=total, replace=True)
        boot_means.append(float(np.mean(sample)) * 100.0)
    mean_pct    = round(float(np.mean(col)) * 100.0, 2)
    ci_low_pct  = round(float(np.percentile(boot_means, 2.5)), 2)
    ci_high_pct = round(float(np.percentile(boot_means, 97.5)), 2)
    boot_data[str(num_idx + 1)] = [ci_low_pct, mean_pct, ci_high_pct]

# ── 9. Random Forest ────────────────────────────────────────────────────────
# Format RF:  {"num_str": prob_pct}  (np. 14.3 = 14.3%)
# Format RF6: [n1, n2, n3, n4, n5, n6]
SPLIT   = int(total * 0.8)
X_feat  = X[:-1]
X_label = X[1:]
X_tr    = X_feat[:SPLIT]
X_te    = X_feat[SPLIT:]

rf_probs = {}
rf_accs  = []

print("Trening Random Forest (49 klasyfikatorow)...")
for num_idx in range(49):
    y_tr = X_label[:SPLIT, num_idx]
    y_te = X_label[SPLIT:, num_idx]

    clf = RandomForestClassifier(n_estimators=50, max_depth=6,
                                 random_state=42, n_jobs=-1,
                                 class_weight='balanced')
    clf.fit(X_tr, y_tr)

    last_draw = X[-1].reshape(1, -1)
    proba     = clf.predict_proba(last_draw)[0]
    classes   = list(clf.classes_)
    prob_1    = float(proba[classes.index(1.0)]) if 1.0 in classes else 0.0

    rf_probs[str(num_idx + 1)] = round(prob_1 * 100.0, 2)
    rf_accs.append(float(clf.score(X_te, y_te)))

rf_avg_acc = round(float(np.mean(rf_accs)) * 100.0, 1)
rf6_sorted = sorted(rf_probs.keys(), key=lambda k: -rf_probs[k])
rf6        = sorted([int(k) for k in rf6_sorted[:6]])
rf6_str    = ", ".join(map(str, rf6))
print(f"RF gotowy. Avg acc: {rf_avg_acc}%. TOP6: {rf6_str}")

# ── 10. Podmiana placeholderow ───────────────────────────────────────────────
template_path = 'template.html'
if not os.path.exists(template_path):
    print(f"BLAD: Brak pliku {template_path}"); sys.exit(1)

with open(template_path, 'r', encoding='utf-8') as f:
    html = f.read()

replacements = {
    '__FREQ__':       json.dumps(freq_list),
    '__REC__':        json.dumps(recent_freq),
    '__DEC__':        json.dumps(decade_data),
    '__LAST20__':     json.dumps(last20_data),
    '__PAIRS__':      json.dumps(top_pairs),
    '__TRIPLES__':    json.dumps(top_triples),
    '__MARKOV__':     json.dumps(markov_data),
    '__CLUSTERS__':   json.dumps(clusters_json),
    '__BOOT__':       json.dumps(boot_data),
    '__RF__':         json.dumps(rf_probs),
    '__RF6__':        json.dumps(rf6),
    '__TOTAL__':      str(total),
    '__SPLIT__':      str(SPLIT),
    '__RFACC__':      str(rf_avg_acc),
    '__RF6STR__':     rf6_str,
    '__LAST_DATE__':  draws[-1]['date'],
    '__DRAWS_ROWS__': draws_rows,
}

missing = []
for placeholder, value in replacements.items():
    if placeholder in html:
        html = html.replace(placeholder, value)
    else:
        missing.append(placeholder)

if missing:
    print(f"OSTRZEZENIE: Nie znaleziono w template: {', '.join(missing)}")

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(html)

print(f"OK — zapisano index.html ({len(html):,} bajtow). "
      f"Ostatnie: {draws[-1]['date']} -> {draws[-1]['numbers']}")
