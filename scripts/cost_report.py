import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from casefile.budget import MAX_DOLLARS
from casefile.db import db


def main():
    conn = db.get_connection()
    rows = conn.execute(
        "SELECT claim_id, SUM(dollars) AS total_dollars FROM trace_events "
        "GROUP BY claim_id ORDER BY claim_id"
    ).fetchall()
    conn.close()

    claim_ids = [r["claim_id"] for r in rows]
    dollars = [float(r["total_dollars"]) for r in rows]

    print(f"{'claim_id':<12}{'dollars':>10}")
    for cid, d in zip(claim_ids, dollars):
        print(f"{cid:<12}{d:>10.5f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(claim_ids, dollars)
    ax.axhline(MAX_DOLLARS, color="red", linestyle="--", label=f"ceiling (${MAX_DOLLARS})")
    ax.set_ylabel("dollars per claim")
    ax.set_xticklabels(claim_ids, rotation=90)
    ax.legend()
    fig.tight_layout()

    os.makedirs("reports", exist_ok=True)
    fig.savefig("reports/cost_per_claim.png")
    print("Saved reports/cost_per_claim.png")


if __name__ == "__main__":
    main()
