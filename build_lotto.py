#!/usr/bin/env python3
"""
LottoAnalityk – build_lotto.py
Pobiera dane losowań Dużego Lotka z mbnet.com.pl, przeprowadza analizę
statystyczną i ML (Random Forest), a następnie wstawia wyniki do template.html
i zapisuje jako index.html.

Autor: K4j745 / LottoAnalityk
"""

import json
import itertools
import warnings
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import requests
import numpy as np
from scipy import stats as sp_stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

# ── Konfiguracja ──────────────────────────────────────────────────────────────
URL = "http://www.mbnet.com.pl/dl.txt"
TEMPLATE_FILE = "template.html"
OUTPUT_FILE = "index.html"
MIN_DRAWS = 100
WINDOW = 10          # okno cech dla RF
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 8
BOOTSTRAP_SAMPLES = 500
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def log(msg: str) -> None:
    print(f"[LottoAnalityk] {msg}", flush=True)


# ── 1. POBIERANIE DANYCH ─────────────────────────────────────────────────────
log("Pobieranie danych z mbnet.com.pl ...")

try:
    resp = requests.get(URL, timeout=45, headers=HEADERS)
    resp.raise_for_status()
    raw_text = resp.text
except Exception as exc:
    log(f"BŁĄD pobierania danych: {exc}")
    sys.exit(1)

log(f"Pobrano {len(raw_text)} bajtów.")

# ── 2. PARSOWANIE ─────────────────────────────────────────────────────────────
draws: list[dict] = []

for line in raw_text.strip().splitlines():
    line = line.strip()
    if not line:
        continue
    # Format: "ID. DD.MM.YYYY N1,N2,N3,N4,N5,N6"
    parts = line.split(maxsplit=2)
    if len(parts) < 3:
        continue
    try:
        draw_id = int(parts[0].rstrip("."))
        date_str = parts[1]
        nums = sorted(int(x) for x in parts[2].split(","))
        if len(nums) != 6 or not all(1 <= n <= 49 for n in nums):
            continue
        draws.append({"id": draw_id, "date": date_str, "numbers": nums})
    except (ValueError, IndexError):
        continue

if len(draws) < MIN_DRAWS:
    log(f"BŁĄD: Za mało losowań ({len(draws)}), wymagane min. {MIN_DRAWS}.")
    sys.exit(1)

# Sortuj chronologicznie po ID (rosnąco)
draws.sort(key=lambda d: d["id"])
total = len(draws)
log(f"Zparsowano {total} losowań (od {draws[0]['date']} do {draws[-1]['date']}).")

# ── 3. ANALIZA STATYSTYCZNA ──────────────────────────────────────────────────

# --- 3a. Częstotliwość wszystkich czasów ---
all_nums = [n for d in draws for n in d["numbers"]]
freq_counter = Counter(all_nums)
# Format: [[1, cnt], [2, cnt], ..., [49, cnt]]
freq_data = [[i, freq_counter.get(i, 0)] for i in range(1, 50)]

# --- 3b. Częstotliwość ostatnie 365 dni ---
try:
    last_date = datetime.strptime(draws[-1]["date"], "%d.%m.%Y")
except ValueError:
    last_date = datetime.now()

cutoff = last_date - timedelta(days=365)
recent_draws = []
for d in draws:
    try:
        dd = datetime.strptime(d["date"], "%d.%m.%Y")
        if dd >= cutoff:
            recent_draws.append(d)
    except ValueError:
        continue

rec_counter = Counter(n for d in recent_draws for n in d["numbers"])
rec_data = [[i, rec_counter.get(i, 0)] for i in range(1, 50)]

# --- 3c. Częstotliwość wg dekad ---
decade_data: dict[str, dict[str, int]] = {}
for d in draws:
    try:
        year = int(d["date"].split(".")[-1])
    except (ValueError, IndexError):
        continue
    dec_key = str((year // 10) * 10)
    if dec_key not in decade_data:
        decade_data[dec_key] = {str(i): 0 for i in range(1, 50)}
    for n in d["numbers"]:
        decade_data[dec_key][str(n)] += 1

# --- 3d. Top 10 par i trójek (Association Rules) ---
log("Obliczanie par i trójek ...")
pair_counter: Counter = Counter()
triple_counter: Counter = Counter()
for d in draws:
    nums = d["numbers"]
    pair_counter.update(itertools.combinations(nums, 2))
    triple_counter.update(itertools.combinations(nums, 3))

# Format: [[[n1,n2], count], ...]
top_pairs = [[list(k), v] for k, v in pair_counter.most_common(10)]
top_triples = [[list(k), v] for k, v in triple_counter.most_common(10)]

# --- 3e. Łańcuch Markowa (top 3 następników) ---
log("Budowanie macierzy Markowa ...")
markov_raw: dict[int, Counter] = defaultdict(Counter)
for i in range(len(draws) - 1):
    for c in draws[i]["numbers"]:
        markov_raw[c].update(draws[i + 1]["numbers"])

# Format: {"1": [[next_num, count], ...top3], ...}
markov_json: dict[str, list] = {}
for num in range(1, 50):
    top3 = markov_raw[num].most_common(3)
    markov_json[str(num)] = [[k, v] for k, v in top3]

# --- 3f. KMeans Clustering (k=5) ---
log("KMeans clustering ...")
freq_vals = np.array([freq_counter.get(i, 0) for i in range(1, 50)]).reshape(-1, 1)
kmeans = KMeans(n_clusters=5, n_init=10, random_state=42).fit(freq_vals)
labels = kmeans.labels_.tolist()

# Format: {"0": [num1, num2, ...], "1": [...], ...}
clusters: dict[str, list[int]] = defaultdict(list)
for i, label in enumerate(labels):
    clusters[str(label)].append(i + 1)
clusters = dict(clusters)

# --- 3g. Bootstrap CI 95% ---
# Procent = (ile razy padła / ile było losowań) * 100
# Bazą jest total_draws, nie total_numbers (żeby skala była ~12% dla fair)
log("Bootstrap confidence intervals ...")
boot_data: dict[str, list[float]] = {}

# Budujemy tablicę per-draw: 1 jeśli liczba padła, 0 jeśli nie
draw_presence = np.zeros((total, 49), dtype=np.float32)
for idx, d in enumerate(draws):
    for n in d["numbers"]:
        draw_presence[idx, n - 1] = 1.0

for num in range(1, 50):
    col = draw_presence[:, num - 1]  # 1/0 per draw
    boot_pcts = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = np.random.choice(col, size=len(col), replace=True)
        boot_pcts.append(float(np.mean(sample) * 100))
    boot_arr = np.array(boot_pcts)
    lo = round(float(np.percentile(boot_arr, 2.5)), 2)
    mean_val = round(float(np.mean(boot_arr)), 2)
    hi = round(float(np.percentile(boot_arr, 97.5)), 2)
    boot_data[str(num)] = [lo, mean_val, hi]

# --- 3h. Low/High split ---
low_count = sum(1 for n in all_nums if n <= 24)
high_count = sum(1 for n in all_nums if n > 24)

# ── 4. MACHINE LEARNING (Random Forest) ─────────────────────────────────────
log("Przygotowywanie danych ML ...")


def prepare_ml_data(draws_list: list[dict], window: int = WINDOW):
    """
    Tworzy macierz cech i etykiet dla 49 binarnych klasyfikatorów.
    Cechy: binarny wektor 49-wymiarowy z `window` ostatnich losowań (spłaszczony)
           + rolling frequency z `window` losowań.
    Etykiety: binarny wektor 49 (czy liczba wypadła w danym losowaniu).
    """
    n = len(draws_list)
    # Binarna macierz obecności: (n_draws, 49)
    presence = np.zeros((n, 49), dtype=np.float32)
    for i, d in enumerate(draws_list):
        for num in d["numbers"]:
            presence[i, num - 1] = 1.0

    X_list, Y_list = [], []
    for i in range(window, n):
        # Cechy: spłaszczony binarny wektor z ostatnich `window` losowań
        flat_history = presence[i - window : i].flatten()
        # + rolling frequency
        rolling_freq = presence[i - window : i].sum(axis=0) / window
        features = np.concatenate([flat_history, rolling_freq])
        X_list.append(features)
        Y_list.append(presence[i])

    return np.array(X_list), np.array(Y_list)


# Użyj max 600 ostatnich losowań dla szybkości (nadal dość danych)
ml_draws = draws[-600:] if len(draws) > 600 else draws
X_all, Y_all = prepare_ml_data(ml_draws)

if len(X_all) < 50:
    log("UWAGA: Za mało danych ML, generuję losowe prawdopodobieństwa.")
    rf_probs_dict = {str(i): round(float(np.random.uniform(8, 18)), 2) for i in range(1, 50)}
    rf6 = list(range(1, 7))
    rf_acc_str = "N/A"
else:
    # Train/test split: ostatnie 20% jako test
    split_idx = int(len(X_all) * 0.8)
    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    Y_train, Y_test = Y_all[:split_idx], Y_all[split_idx:]

    log(f"Trenowanie 49 klasyfikatorów RF (train={len(X_train)}, test={len(X_test)}) ...")

    rf_probs_raw = np.zeros(49)
    accuracies = []

    for i in range(49):
        clf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, Y_train[:, i])

        # Accuracy na test set
        acc = clf.score(X_test, Y_test[:, i])
        accuracies.append(acc)

        # Predict proba na OSTATNIM wierszu (=najnowsze dane → predykcja na następne losowanie)
        last_features = X_all[-1:].copy()
        proba = clf.predict_proba(last_features)
        # proba[0] = [P(0), P(1)] — chcemy P(1), czyli że liczba wypadnie
        if proba.shape[1] == 2:
            rf_probs_raw[i] = proba[0, 1]
        elif proba.shape[1] == 1:
            # Jeśli model widział tylko jedną klasę
            rf_probs_raw[i] = proba[0, 0] if clf.classes_[0] == 1 else 0.0
        else:
            rf_probs_raw[i] = 0.0

    avg_acc = float(np.mean(accuracies)) * 100
    rf_acc_str = f"{avg_acc:.1f}"

    # Normalizuj prawdopodobieństwa na procenty (skaluj do sensownego zakresu)
    rf_probs_pct = rf_probs_raw * 100
    rf_probs_dict = {str(i + 1): round(float(rf_probs_pct[i]), 2) for i in range(49)}

    # Top 6 wg prawdopodobieństwa
    rf6 = (np.argsort(rf_probs_raw)[-6:][::-1] + 1).tolist()
    rf6.sort()

    log(f"RF gotowy. Avg accuracy: {rf_acc_str}%, TOP6: {rf6}")

# ── 5. PRZYGOTOWANIE DANYCH DO WSTAWIENIA ────────────────────────────────────

# Ostatnie 20 losowań – dane JSON + wiersze HTML tabeli
last20_data = []
last20_draws = draws[-20:]
draws_rows_html = ""

for d in reversed(last20_draws):
    last20_data.append({"id": d["id"], "date": d["date"], "numbers": d["numbers"]})
    balls_html = "".join(
        f'<div class="dball">{n}</div>' for n in d["numbers"]
    )
    draws_rows_html += (
        f'<tr>'
        f'<td style="color:var(--txm);font-variant-numeric:tabular-nums">{d["id"]}</td>'
        f'<td style="white-space:nowrap">{d["date"]}</td>'
        f'<td><div class="dnums">{balls_html}</div></td>'
        f'</tr>'
    )

# Odwróć last20_data z powrotem do chronologicznego (najstarsze → najnowsze)
last20_data.reverse()

last_date_str = draws[-1]["date"]
rf6_str = ", ".join(str(n) for n in sorted(rf6))
split_json = json.dumps({"low": low_count, "high": high_count})

# ── 6. PODMIANA PLACEHOLDERÓW W SZABLONIE ────────────────────────────────────
log("Generowanie index.html ...")

try:
    with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
        html = f.read()
except FileNotFoundError:
    log(f"BŁĄD: Nie znaleziono pliku {TEMPLATE_FILE}.")
    sys.exit(1)

replacements = {
    "__FREQ__": json.dumps(freq_data),
    "__REC__": json.dumps(rec_data),
    "__DEC__": json.dumps(decade_data),
    "__LAST20__": json.dumps(last20_data),
    "__PAIRS__": json.dumps(top_pairs),
    "__TRIPLES__": json.dumps(top_triples),
    "__MARKOV__": json.dumps(markov_json),
    "__CLUSTERS__": json.dumps(clusters),
    "__BOOT__": json.dumps(boot_data),
    "__RF__": json.dumps(rf_probs_dict),
    "__RF6__": json.dumps(sorted(rf6)),
    "__TOTAL__": str(total),
    "__SPLIT__": str(total - 300),       # Train set size for display
    "__RFACC__": rf_acc_str,
    "__RF6STR__": rf6_str,
    "__LAST_DATE__": last_date_str,
    "__DRAWS_ROWS__": draws_rows_html,
}

for placeholder, value in replacements.items():
    count = html.count(placeholder)
    if count == 0:
        log(f"UWAGA: Placeholder {placeholder} nie znaleziony w szablonie.")
    html = html.replace(placeholder, value)

try:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)
except Exception as exc:
    log(f"BŁĄD zapisu {OUTPUT_FILE}: {exc}")
    sys.exit(1)

log(f"Sukces! Wygenerowano {OUTPUT_FILE} ({len(html)} bajtów).")
log(f"Statystyki: {total} losowań, {len(recent_draws)} z ostatniego roku, RF6={rf6}")
