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
    for pai
