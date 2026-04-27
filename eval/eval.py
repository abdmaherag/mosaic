"""Retrieval evaluation script with comparison mode.

Usage:
    # Standard eval (hybrid + multi-query, the default pipeline)
    python eval.py

    # Compare vector-only vs hybrid+multi-query retrieval
    python eval.py --compare

    # Custom options
    python eval.py --api http://localhost:8000 --dataset eval_dataset.json --k 5

Metrics reported:
    hit@k       — fraction of questions where expected_document appears in top-k citations
    keyword@k   — fraction of questions where >=1 expected keyword appears in top-k chunk text
"""

import argparse
import json
import sys
import urllib.request


DEVICE_ID = "eval-device-0001"


def query_api(api_base: str, question: str, endpoint: str = "/query/") -> list[dict]:
    """Call a query endpoint and return citations."""
    url = f"{api_base}{endpoint}"
    data = json.dumps({"question": question}).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "X-Device-ID": DEVICE_ID},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())
    return body.get("citations", [])


def run_eval(api_base: str, pairs: list[dict], k: int, endpoint: str, label: str) -> dict:
    """Run evaluation on a set of pairs and return metrics."""
    hit_count = 0
    kw_count = 0
    total = 0

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    for pair in pairs:
        question = pair["question"]
        expected_doc = pair.get("expected_document", "").lower()
        expected_kws = [kw.lower() for kw in pair.get("expected_keywords", [])]

        print(f"\nQ: {question}")
        total += 1
        try:
            citations = query_api(api_base, question, endpoint)[:k]
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        docs_returned = [c["document"].lower() for c in citations]
        text_blob = " ".join(c["chunk_text"].lower() for c in citations)

        hit = expected_doc in docs_returned if expected_doc else True
        kw_hit = any(kw in text_blob for kw in expected_kws) if expected_kws else True

        if hit:
            hit_count += 1
        if kw_hit:
            kw_count += 1

        print(f"  hit@{k}: {'YES' if hit else 'NO'} | keyword@{k}: {'YES' if kw_hit else 'NO'}")
        print(f"  returned: {docs_returned}")

    if total == 0:
        print("\nNo questions evaluated.")
        return {"hit": 0, "keyword": 0, "total": 0}

    print(f"\n--- {label} Results ({total} questions, k={k}) ---")
    print(f"hit@{k}:     {hit_count}/{total} = {hit_count/total:.1%}")
    print(f"keyword@{k}: {kw_count}/{total} = {kw_count/total:.1%}")

    return {"hit": hit_count, "keyword": kw_count, "total": total}


def evaluate(api_base: str, dataset_path: str, k: int) -> None:
    with open(dataset_path) as f:
        dataset = json.load(f)

    pairs = dataset.get("pairs", [])
    if not pairs:
        print("No evaluation pairs found in dataset.")
        sys.exit(1)

    run_eval(api_base, pairs, k, "/query/", "Hybrid + Multi-Query (default pipeline)")


def compare(api_base: str, dataset_path: str, k: int) -> None:
    """Run both vector-only and hybrid+multi-query, then compare."""
    with open(dataset_path) as f:
        dataset = json.load(f)

    pairs = dataset.get("pairs", [])
    if not pairs:
        print("No evaluation pairs found in dataset.")
        sys.exit(1)

    vector_results = run_eval(
        api_base, pairs, k, "/query/vector-only",
        "Vector-Only Retrieval (baseline)",
    )
    hybrid_results = run_eval(
        api_base, pairs, k, "/query/",
        "Hybrid + Multi-Query (RRF pipeline)",
    )

    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")

    v_total = vector_results["total"] or 1
    h_total = hybrid_results["total"] or 1

    v_hit = vector_results["hit"] / v_total
    h_hit = hybrid_results["hit"] / h_total
    v_kw = vector_results["keyword"] / v_total
    h_kw = hybrid_results["keyword"] / h_total

    print(f"\n{'Metric':<20} {'Vector-Only':>15} {'Hybrid+MQ':>15} {'Delta':>10}")
    print(f"{'-'*60}")
    print(f"{'hit@' + str(k):<20} {v_hit:>14.1%} {h_hit:>14.1%} {h_hit - v_hit:>+9.1%}")
    print(f"{'keyword@' + str(k):<20} {v_kw:>14.1%} {h_kw:>14.1%} {h_kw - v_kw:>+9.1%}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--dataset", default="eval_dataset.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--compare", action="store_true", help="Compare vector-only vs hybrid+multi-query")
    args = parser.parse_args()

    if args.compare:
        compare(args.api, args.dataset, args.k)
    else:
        evaluate(args.api, args.dataset, args.k)
