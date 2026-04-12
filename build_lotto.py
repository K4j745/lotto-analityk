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
    print("BŁĄD: timeout przy pobieraniu danych (>30s)")
    sys.exit(1)
except requests.exceptions.ConnectionError as e:
    print(f"BŁĄD połączenia: {e}")
    sys.exit(1)
except requests.exceptions.HTTPError as e:
    print(f"BŁĄD HTTP: {e}")
    sys.exit(1)
except Exception as e:
    print(f"BŁĄD nieoczekiwany przy pobieraniu: {e}")
    sys.exit(1)

draws = []
for line in r.text.strip().split('\n'):
    parts = line.strip().split(' ', 2)
    if len(parts) == 3:
        try:
            nums = sorted(map(int, parts[2].split(',')))
            if len(nums) == 6 and all(1 <= n <= 49 for n in nums):
                draws.append({"id": int(parts[0].rstrip('.')), "date": parts[1], "numbers": nums})
        except (ValueError, IndexError):
            continue

total = len(draws)
print(f"Zaladowano {total} losowan.")

if total < 100:
    print(f"BŁĄD: za mało losowań ({total}), dane prawdopodobnie uszkodzone.")
    sys.exit(1)

print(f"Ostatnie: {draws[-1]['date']} -> {draws[-1]['numbers']}")

freq = Counter()
for d in draws: freq.update(d['numbers'])
freq_list = [[i, freq.get(i,
