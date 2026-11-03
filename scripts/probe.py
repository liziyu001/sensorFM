"""Run a frozen-encoder linear probe from a saved embedding file.

Expected .npz keys:
    embeddings:   [N, d] day-level or token-level embeddings
    subject_id:   [N]    subject id per row
    labels_<name>: [n_subjects] person-level labels

Example:
    uv run python scripts/probe.py --npz data/embeddings.npz \
        --label stress_level --task regression
"""

from __future__ import annotations

import argparse

import numpy as np

from sensorfm.eval.probe import linear_probe, person_aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--label", default="stress_level")
    parser.add_argument("--task", default="regression", choices=["classification", "regression"])
    parser.add_argument("--components", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.npz)
    key = f"labels_{args.label}"
    if key not in data:
        raise KeyError(f"{key!r} not found; available: {sorted(data.files)}")

    embeddings = data["embeddings"]
    subject_id = data["subject_id"]
    subjects = np.unique(subject_id)
    labels = data[key]

    if labels.shape[0] != subjects.size:
        raise ValueError(f"expected {subjects.size} person-level labels, got {labels.shape[0]}")

    person = person_aggregate(embeddings, subject_id)
    result = linear_probe(
        person,
        labels,
        task_type=args.task,
        n_components=args.components,
    )
    print(result)


if __name__ == "__main__":
    main()
