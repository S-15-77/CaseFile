import argparse
import json

from casefile.db import db
from casefile.pipeline import process_claim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims-file", default="data/synthetic_claims.json")
    parser.add_argument("--auto-approve", action="store_true")
    args = parser.parse_args()

    with open(args.claims_file) as f:
        claims = json.load(f)

    conn = db.get_connection()
    db.init_schema(conn)

    print(f"Running {len(claims)} claims...")
    for claim_dict in claims:
        try:
            result = process_claim(conn, claim_dict, auto_approve=args.auto_approve)
            print(f"{claim_dict['claim_id']}: {result.status.value} "
                  f"(steps={result.step_count}, dollars=${result.dollars_used:.5f}, "
                  f"revisions={result.revision_count})")
        except Exception as exc:
            print(f"{claim_dict['claim_id']}: ERRORED ({exc})")

    conn.close()


if __name__ == "__main__":
    main()
