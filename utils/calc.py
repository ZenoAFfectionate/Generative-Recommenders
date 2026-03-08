import os
import re
import fire
import math
import json
import numpy as np
from collections import Counter
from tqdm import tqdm


def parse_sid(sid):
    """Parse SID like '<a_236><b_231><c_226>' into (a, b, c) tuple."""
    parts = re.findall(r'<([abc])_(\d+)>', sid)
    return tuple(int(v) for _, v in parts)


def sid_similarity(sid1, sid2):
    """Compute hierarchical similarity between two SIDs.
    Shared prefix levels / total levels. Returns 0, 1/3, 2/3, or 1."""
    p1, p2 = parse_sid(sid1), parse_sid(sid2)
    if not p1 or not p2:
        return 0.0
    shared = 0
    for v1, v2 in zip(p1, p2):
        if v1 == v2:
            shared += 1
        else:
            break
    return shared / len(p1)


def gao(path, item_path, train_path=None):
    if type(path) != list:
        path = [path]
    if item_path.endswith(".txt"):
        item_path = item_path[:-4]
    CC = 0

    f = open(f"{item_path}.txt", 'r')
    items = f.readlines()
    f.close()
    item_names = [_.split('\t')[0].strip() for _ in items]
    item_ids = [_ for _ in range(len(item_names))]
    item_dict = dict()
    for i in range(len(item_names)):
        if item_names[i] not in item_dict:
            item_dict[item_names[i]] = [item_ids[i]]
        else:
            item_dict[item_names[i]].append(item_ids[i])

    total_items = len(item_names)
    item_set = set(item_names)

    # Load item popularity from training data (optional, for novelty/long-tail metrics)
    item_popularity = None
    if train_path:
        import pandas as pd
        import ast
        train_df = pd.read_csv(train_path)
        pop_counter = Counter(train_df['item_sid'].tolist())
        for h in train_df['history_item_sid']:
            try:
                sids = ast.literal_eval(h)
                pop_counter.update(sids)
            except Exception:
                pass
        total_interactions = sum(pop_counter.values())
        item_popularity = {sid: count / total_interactions for sid, count in pop_counter.items()}

    topk_list = [1, 3, 5, 10, 20, 50]
    n_beam = -1

    for p in path:
        f = open(p, 'r')
        test_data = json.load(f)
        f.close()

        text = [[_.strip("\"\n").strip() for _ in sample["predict"]] for sample in test_data]
        n_samples = len(text)

        for idx, sample in tqdm(enumerate(text)):
            if n_beam == -1:
                n_beam = len(sample)
                valid_topk = [k for k in topk_list if k <= n_beam]
                n_valid = len(valid_topk)
                ALLNDCG = np.zeros(n_valid)
                ALLHR = np.zeros(n_valid)
                ALLMRR = np.zeros(n_valid)
                # Coverage: collect unique items recommended across all users
                coverage_sets = [set() for _ in range(n_valid)]
                # ILS accumulators
                ALLILS = np.zeros(n_valid)
                # Novelty accumulators
                ALLNOVELTY = np.zeros(n_valid)
                novelty_counts = np.zeros(n_valid)
                # Gini: collect all predicted items per topk for frequency distribution
                gini_lists = [[] for _ in range(n_valid)]

            if type(test_data[idx]['output']) == list:
                target_item = test_data[idx]['output'][0].strip("\"").strip(" ")
            else:
                target_item = test_data[idx]['output'].strip(" \n\"")

            # Find rank of target
            minID = 1000000
            for i in range(len(sample)):
                if sample[i] not in item_dict:
                    CC += 1
                if sample[i] == target_item:
                    minID = i
                    break

            for ti, topk in enumerate(valid_topk):
                # HR & NDCG
                if minID < topk:
                    ALLNDCG[ti] += 1 / math.log(minID + 2)
                    ALLHR[ti] += 1
                    ALLMRR[ti] += 1.0 / (minID + 1)

                top_items = sample[:topk]

                # Coverage
                coverage_sets[ti].update(top_items)

                # Gini
                gini_lists[ti].extend(top_items)

                # ILS (Intra-List Similarity via SID prefix overlap)
                if topk <= 50:  # skip very large K for O(K^2) computation
                    pair_sim = 0.0
                    n_pairs = 0
                    for a in range(len(top_items)):
                        for b in range(a + 1, len(top_items)):
                            pair_sim += sid_similarity(top_items[a], top_items[b])
                            n_pairs += 1
                    if n_pairs > 0:
                        ALLILS[ti] += pair_sim / n_pairs

                # Novelty (Mean Self-Information)
                if item_popularity:
                    for item in top_items:
                        pop = item_popularity.get(item, 1e-9)
                        ALLNOVELTY[ti] += -math.log2(pop)
                        novelty_counts[ti] += 1

        # ============ Print Results ============
        print(f"\n{'='*60}")
        print(f"Results for: {p}")
        print(f"n_beam={n_beam}, samples={n_samples}, total_items={total_items}")
        print(f"Invalid predictions (not in item set): {CC}")
        print(f"{'='*60}")

        # --- Accuracy Metrics ---
        print(f"\n--- Accuracy Metrics ---")
        header = "\t".join([f"@{k}" for k in valid_topk])
        ndcg_vals = ALLNDCG / n_samples / (1.0 / math.log(2))
        hr_vals = ALLHR / n_samples
        mrr_vals = ALLMRR / n_samples
        print(f"K\t{header}")
        print(f"NDCG\t" + "\t".join([f"{v:.4f}" for v in ndcg_vals]))
        print(f"HR\t" + "\t".join([f"{v:.4f}" for v in hr_vals]))
        print(f"MRR\t" + "\t".join([f"{v:.4f}" for v in mrr_vals]))

        # --- Diversity Metrics ---
        print(f"\n--- Diversity Metrics ---")
        coverage_vals = [len(coverage_sets[ti]) / total_items for ti in range(len(valid_topk))]
        ils_vals = ALLILS / n_samples
        print(f"Coverage\t" + "\t".join([f"{v:.4f}" for v in coverage_vals]))
        print(f"ILS\t" + "\t".join([f"{v:.4f}" for v in ils_vals]))

        # Entropy of prediction distribution per topk
        entropy_vals = []
        for ti in range(len(valid_topk)):
            freq = Counter(gini_lists[ti])
            total = len(gini_lists[ti])
            probs = np.array([c / total for c in freq.values()])
            entropy = -np.sum(probs * np.log2(probs + 1e-12))
            entropy_vals.append(entropy)
        max_entropy = math.log2(total_items)
        print(f"Entropy\t" + "\t".join([f"{v:.4f}" for v in entropy_vals]))
        print(f"  (max possible entropy: {max_entropy:.4f})")

        # --- Fairness Metrics ---
        print(f"\n--- Fairness Metrics ---")
        gini_vals = []
        for ti in range(len(valid_topk)):
            freq = Counter(gini_lists[ti])
            counts = np.array(sorted(freq.values()))
            n = len(counts)
            if n == 0:
                gini_vals.append(0.0)
                continue
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * counts) - (n + 1) * np.sum(counts)) / (n * np.sum(counts))
            gini_vals.append(gini)
        print(f"Gini\t" + "\t".join([f"{v:.4f}" for v in gini_vals]))

        # Long-tail coverage (only if train_path provided)
        if item_popularity:
            median_pop = np.median(list(item_popularity.values()))
            longtail_items = {sid for sid, pop in item_popularity.items() if pop <= median_pop}
            longtail_coverage = []
            for ti in range(len(valid_topk)):
                predicted_longtail = set(gini_lists[ti]) & longtail_items
                if len(longtail_items) > 0:
                    longtail_coverage.append(len(predicted_longtail) / len(longtail_items))
                else:
                    longtail_coverage.append(0.0)
            print(f"LongTail-Cov\t" + "\t".join([f"{v:.4f}" for v in longtail_coverage]))
            print(f"  (long-tail items: {len(longtail_items)}, threshold: pop <= {median_pop:.6f})")

        # --- Novelty Metrics ---
        if item_popularity:
            print(f"\n--- Novelty Metrics ---")
            novelty_vals = ALLNOVELTY / (novelty_counts + 1e-12)
            print(f"Novelty\t" + "\t".join([f"{v:.4f}" for v in novelty_vals]))
            print(f"  (Mean Self-Information in bits; higher = more novel)")


if __name__ == '__main__':
    fire.Fire(gao)
