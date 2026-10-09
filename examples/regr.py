import sqlite3
import numpy as np

conn = sqlite3.connect("output/egfr.sqlite")

rows = conn.execute("""
    SELECT docking_score, gbsa_score
    FROM compound
    WHERE docking_score IS NOT NULL
      AND gbsa_score IS NOT NULL
""").fetchall()

x = np.array([r[0] for r in rows])
y = np.array([r[1] for r in rows])

# y = a*x + b
a, b = np.polyfit(x, y, 1)

y_pred = a * x + b

ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - ss_res / ss_tot

r = np.corrcoef(x, y)[0, 1]

print(f"N = {len(x)}")
print(f"gbsa_score = {a:.4f} * docking_score + {b:.4f}")
print(f"Pearson r = {r:.4f}")
print(f"R² = {r2:.4f}")
print( "----------------------------------------" )

for iteration in range(100):
    rows = conn.execute("""
        SELECT docking_score, gbsa_score
        FROM compound
        WHERE iteration = ?
          AND docking_score IS NOT NULL
          AND gbsa_score IS NOT NULL
    """, (iteration,)).fetchall()

    if len(rows) < 3:
        continue

    x = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])

    a, b = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]

    print(
        f"{iteration:2d}  "
        f"N={len(x):5d}  "
        f"slope={a:7.3f}  "
        f"r={r:6.3f}  "
        f"R²={r*r:6.3f}"
    )

