import argparse

from casefile.db import db


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("claim_id")
    args = parser.parse_args()

    conn = db.get_connection()
    events = db.list_trace_events(conn, args.claim_id)
    conn.close()

    if not events:
        print(f"No trace events for {args.claim_id}")
        return

    print(f"Node path for {args.claim_id}:")
    for e in events:
        print(f"  step {e['step_number']}: {e['node_name']} "
              f"(tokens_in={e['tokens_in']}, dollars=${float(e['dollars']):.5f}, status={e['status']})")


if __name__ == "__main__":
    main()
