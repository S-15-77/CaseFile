import argparse

from casefile.replay import replay_claim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("claim_id")
    args = parser.parse_args()

    result = replay_claim(args.claim_id)
    print(f"Replayed {args.claim_id}: status={result.status.value}, "
          f"history={result.history}")


if __name__ == "__main__":
    main()
