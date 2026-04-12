import json, itertools, random, warnings, os, requests, sys
warnings.filterwarnings('ignore')
from collections import Counter, defaultdict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

print("Pobieram dane z mbnet.com.pl...")
try:
    r = requests.get('http://www.mbnet.com.pl/dl.txt', timeout=30)
    r.raise_for_status()
except requests.exceptions.Timeout:
    print("BLAD: timeout przy pobieraniu danych (>30s)"); sys.exit(1)
except requests.exceptions.ConnectionError as e:
    print(f"BLAD polaczenia: {e}"); sys.exit(1)
except requests.exceptions.HTTPError as e:
    print(f"BLAD HTTP: {e}"); sys.exit(1)
except Exception as e:
    print(f"BLAD nieoczekiwany: {e}"); sys.exit(1)

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
    print(f"BLAD: za malo losowan ({total}), dane prawdopodobnie uszkodzone."); sys.exit(1)
print(f"Zaladowano {total} losowan. Ostatnie: {draws[-1]['date']} -> {draws[-1]['numbers']}")

# --- Czestotliwosci ---
freq = Counter()
for d in draws:
    freq.update(d['numbers'])
freq_list = [[i, freq.get(i, 0)] for i in range(1, 50)]

recent = draws[-365:]
rfq = Counter()
for d in recent:
    rfq.update(d['numbers'])
recent_freq = [[i, rfq.get(i, 0)] for i in range(1, 50)]

decade_freq = {}
for d in draws:
    year = int(d['date'].split('.')[-1])
    dec = (year // 10) * 10
    if dec not in decade_freq:
        decade_freq[dec] = Counter()
    decade_freq[dec].update(d['numbers'])
decade_data = {str(dec): {str(i): cnt.get(i, 0) for i in range(1, 50)} for dec, cnt in decade_freq.items()}

# --- Ostatnie 20 losowan ---
last20_data = [{"date": d['date'], "numbers": d['numbers']} for d in draws[-20:]]

# --- Tabela losowan (ostatnie 200) ---
draws_rows = ""
for d in reversed(draws[-200:]):
    nums_html = " ".join(f'<span class="ball">{n}</span>' for n in d['numbers'])
    draws_rows += f'<tr><td>{d["id"]}</td><td>{d["date"]}</td><td>{nums_html}</td></tr>\n'

# --- Pary i trojki (Association Rules) ---
pair_count = Counter()
triple_count = Counter()
for d in draws:
    for pair in itertools.combinations(d['numbers'], 2):
        pair_count[pair] += 1
    for triple in itertools.combinations(d['numbers'], 3):
        triple_count[triple] += 1

top_pairs = [{"pair": list(p), "count": c} for p, c in pair_count.most_common(20)]
top_triples = [{"triple": list(t), "count": c} for t, c in triple_count.most_common(10)]

# --- Lancuch Markowa ---
markov = defaultdict(Counter)
for d in draws:
    nums = d['numbers']
    for a in nums:
        for b in nums:
            if a != b:
                markov[a][b] += 1
markov_data = {str(n): {str(k): v for k, v in markov[n].most_common(5)} for n in range(1, 50)}

# --- Macierz binarnych cech (draws x 49) ---
X = np.zeros((total, 49), dtype=np.float32)
for i, d in enumerate(draws):
    for n in d['numbers']:
        X[i, n - 1] = 1.0

# --- KMeans Clustering k=5 ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
cluster_data = defaultdict(list)
for i in range(max(0, total - 200), total):
    cluster_data[int(labels[i])].append(draws[i]['numbers'])
clusters_json = {str(k): v[-5:] for k, v in cluster_data.items()}

# --- Bootstrap CI 95% (srednia wylosowanej liczby) ---
rng = np.random.default_rng(42)
all_nums = np.array([n for d in draws for n in d['numbers']])
boot_means = [float(np.mean(rng.choice(all_nums, size=len(all_nums), replace=True))) for _ in range(1000)]
boot_ci = {
    "mean": float(np.mean(all_nums)),
    "ci_low": float(np.percentile(boot_means, 2.5)),
    "ci_high": float(np.percentile(boot_means, 97.5))
}

# --- Random Forest (49 klasyfikatorow binarnych) ---
SPLIT = 0.8
# Uzyj przesunietej ramki: cechy = losowanie[t], etykieta = losowanie[t+1]
X_feat = X[:-1]   # wszystkie losowania oprocz ostatniego
X_label = X[1:]   # przesuniete o 1 do przodu

split_idx = int(len(X_feat) * SPLIT)
X_tr, X_te = X_feat[:split_idx], X_feat[split_idx:]

rf_probs = []
rf_accs = []

print("Trening Random Forest (49 klasyfikatorow)...")
for num_idx in range(49):
    y_tr = X_label[:split_idx, num_idx]
    y_te = X_label[split_idx:, num_idx]

    clf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)

    # Predykcja prawdopodobienstwa dla KOLEJNEGO losowania na podstawie ostatniego
    last_draw = X[-1].reshape(1, -1)
    prob_classes = clf.predict_proba(last_draw)[0]
    if len(clf.classes_) == 2:
        rf_probs.append(float(prob_classes[1]))
    else:
        # Tylko jedna klasa w danych treningowych
        rf_probs.append(float(clf.classes_[0]))

    rf_accs.append(float(clf.score(X_te, y_te)))

rf_data = [[i + 1, round(rf_probs[i], 4)] for i in range(49)]
rf_sorted_idx = sorted(range(49), key=lambda i: -rf_probs[i])
rf6 = sorted([i + 1 for i in rf_sorted_idx[:6]])
rf6_str = ", ".join(map(str, rf6))
rf_acc = round(float(np.mean(rf_accs)), 4)

print(f"RF gotowy. Srednia dokladnosc: {rf_acc:.4f}. TOP6: {rf6_str}")

# --- Podmiana placeholderow w szablonie ---
template_path = 'template.html'
if not os.path.exists(template_path):
    print(f"BLAD: Brak pliku {template_path}"); sys.exit(1)

with open(template_path, 'r', encoding='utf-8') as f:
    html = f.read()

replacements = {
    '__FREQ__':      json.dumps(freq_list),
    '__REC__':       json.dumps(recent_freq),
    '__DEC__':       json.dumps(decade_data),
    '__LAST20__':    json.dumps(last20_data),
    '__PAIRS__':     json.dumps(top_pairs),
    '__TRIPLES__':   json.dumps(top_triples),
    '__MARKOV__':    json.dumps(markov_data),
    '__CLUSTERS__':  json.dumps(clusters_json),
    '__BOOT__':      json.dumps(boot_ci),
    '__RF__':        json.dumps(rf_data),
    '__RF6__':       json.dumps(rf6),
    '__TOTAL__':     str(total),
    '__SPLIT__':     str(split_idx),
    '__RFACC__':     str(rf_acc),
    '__RF6STR__':    rf6_str,
    '__LAST_DATE__': draws[-1]['date'],
    '__DRAWS_ROWS__': draws_rows,
}

missing = []
for placeholder, value in replacements.items():
    if placeholder in html:
        html = html.replace(placeholder, value)
    else:
        missing.append(placeholder)

if missing:
    print(f"OSTRZEZENIE: Nie znaleziono w template.html: {', '.join(missing)}")

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(html)

print(f"Zapisano index.html ({len(html):,} bajtow). Ostatnie losowanie: {draws[-1]['date']}")
