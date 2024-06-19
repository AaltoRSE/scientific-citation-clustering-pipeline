import argparse
import json
import pickle
import re
import time

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, OPTICS
from sklearn.metrics.pairwise import pairwise_distances

REFERENCES_COLUMN = "References"
REFERENCES_AUGMENTED_COLUMN = "References augmented"

def extract_references(data):
    """Return list of references from DataFrame."""
    references = []
    for refs in data[REFERENCES_COLUMN]:
        references.extend(refs.split(";"))
    references = [ref.strip() for ref in references]

    return references


def get_embedding_filename(filename, embedding_model_name):
    stem = Path(filename).stem
    embedding_model_name = embedding_model_name.replace("/", "_")
    filename = f"{stem}___embedding_model={embedding_model_name}___type=embeddings.pkl"
    return filename


def get_clustering_filename(
    filename, embedding_model_name, distance_threshold, type_, suffix
):
    stem = Path(filename).stem
    embedding_model_name = embedding_model_name.replace("/", "_")
    filename = f"{stem}___embedding_model={embedding_model_name}___distance_threshold={distance_threshold}___type={type_}{suffix}"
    return filename


def force_same_year(references, distances):
    # Extract years
    reference_to_year = {}
    for ref in set(references):
        m = re.search(
            r"\((\d{4})[a-z]?\)", ref
        )  # find year in parentheses (e.g. (2021)), drop any letter after the year (e.g. 2021a -> 2021)
        year = m.group(1) if m else None
        reference_to_year[ref] = year

    # Set distance to 1.0 (= similarity 0.0) if references have different years
    for i, ref1 in enumerate(references):
        for j, ref2 in enumerate(references):
            if not reference_to_year[ref1] or not reference_to_year[ref2]:
                continue
            if reference_to_year[ref1] != reference_to_year[ref2]:
                distances[i, j] = 1.0

    return distances


def sort_references(references):
    """Order by length from longest to shortest and then alphabetically"""
    return sorted(references, key=lambda x: (-len(x), x))


def get_codebook(clustering, references):
    # Create inverse clustering: cluster_index => reference_indexes
    inv_clustering = defaultdict(list)
    for i, ref in enumerate(clustering):
        inv_clustering[ref].append(i)

    # Name cluster indexes (after alphabetically first reference in cluster)
    cluster_names = {}
    for cluster_index, references_indexes in inv_clustering.items():
        references_in_cluster = [references[i] for i in references_indexes]
        first_reference = sort_references(references_in_cluster)[0]
        cluster_names[cluster_index] = first_reference

    # Create codebook: reference => cluster
    codebook = defaultdict(list)
    for cluster_index, references_indexes in inv_clustering.items():
        for ref_index in references_indexes:
            codebook[references[ref_index]] = cluster_names[cluster_index]

    # Inverse codebook: cluster => references
    inv_codebook = defaultdict(list)
    for ref, cluster in codebook.items():
        inv_codebook[cluster].append(ref)

    # Sort references in clusters
    for cluster, references in inv_codebook.items():
        inv_codebook[cluster] = sort_references(references)

    return codebook, inv_codebook

def set_diagonal_to(matrix, value):
    for i in range(len(matrix)):
        matrix[i, i] = value
    return matrix


def extract_singleton_indexes_low_memory(distances, distance_threshold):
    """
    References with distance > threshold to all other references are singleton clusters.
    """
    singleton_indexes = []
    for i in range(distances.shape[0]):
        row = distances[i, :]
        row[i] = 1.0
        if np.min(row) > distance_threshold:
            singleton_indexes.append(i)

    return singleton_indexes


def main():
    print("Start")

    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "input_file",
        type=str,
        help="Input CSV file with column 'References'. Mandatory",
    )
    argparser.add_argument(
        "--embedding_model_name",
        type=str,
        default="Snowflake/snowflake-arctic-embed-l",
        help="Model name. Optional",
    )
    argparser.add_argument(
        "--distance_threshold",
        type=float,
        default=0.05,
        help="Distance threshold (0.0 - 1.0). Optional",
    )
    argparser.add_argument(
        "--force_same_year",
        action="store_true",
        help="Force same year within cluster. Optional",
    )
    args = argparser.parse_args()

    # Load CSV data
    # =
    data = pd.read_csv(args.input_file)
    data["References"] = data["References"].fillna("")
    print(f"Read {len(data)} lines from input file: {args.input_file}")

    # Extract references
    # ==================
    references = extract_references(data)
    print(
        f"Read {len(references)} references ({len(references)/len(data):.1f} per article)"
    )

    # Remove duplicates
    references = list(set(references))
    print(
        f"Read {len(references)} unique references ({len(references)/len(data):.1f} per article)"
    )

    # Create output dir
    output_dir = Path(args.input_file).parent / "results/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Embeddings file
    # ===============
    embeddings_file = get_embedding_filename(args.input_file, args.embedding_model_name)
    embeddings_file = output_dir / embeddings_file
    if not Path(embeddings_file).exists():
        # Load embedding model
        print(f"Load embedding model: {args.embedding_model_name}")
        model = SentenceTransformer(args.embedding_model_name)

        # Compute embeddings
        print("Compute embeddings")
        t = time.time()
        embeddings = model.encode(references)
        print(
            f".. Computed {len(references)} embeddings in {time.time()-t:.2f} seconds ({(time.time()-t)/len(references):.3f} seconds per reference)"
        )

        # Write embeddings to file
        print("Write embeddings to file")
        pickle.dump(
            (references, embeddings, args.embedding_model_name),
            open(embeddings_file, "wb"),
        )

        # Done
        print(
            f"Wrote {len(references)} embeddings to {Path(embeddings_file).absolute()}"
        )

    else:
        # Load embeddings
        print(f"Load embeddings from existing file: {embeddings_file}")
        references, embeddings, embedding_model_name = pickle.load(
            open(embeddings_file, "rb")
        )

    # Clustering
    # ==========
    print("Compute clusters")

    t = time.time()

    # Create codebooks
    # - codebook: reference => cluster name
    # - inverse codebook: cluster name => references
    codebook = {}
    inv_codebook = defaultdict(list)

    # Compute pairwise distances
    metric = "cosine"
    print(f".. Compute pairwise distances (metric: {metric})")
    distances = pairwise_distances(embeddings, embeddings, metric=metric, n_jobs=None)

    # Singleton clusters
    print(".. Create singleton clusters")
    singleton_indexes = extract_singleton_indexes_low_memory(distances, args.distance_threshold)

    for i in singleton_indexes:
        codebook[references[i]] = references[i]
        inv_codebook[references[i]] = [references[i]]

    distances = np.delete(distances, singleton_indexes, axis=0)
    distances = np.delete(distances, singleton_indexes, axis=1)

    references = [ref for i, ref in enumerate(references) if i not in singleton_indexes]

    # Compute clusters
    algo = AgglomerativeClustering(
        metric="precomputed",
        distance_threshold=args.distance_threshold,
        n_clusters=None,
        linkage="average",
    )  # "complete" = highest precision, "single" = highest recall, average = "compromise"
    print(f".. Run clustering algorithm: {algo}")
    clustering = algo.fit(distances)

    print(f".. Took {time.time()-t:.2f} seconds")

    # Expand codebook
    print(".. Update codebook")
    codebook_new, inv_codebook_new = get_codebook(clustering.labels_, references)
    codebook.update(codebook_new)
    inv_codebook.update(inv_codebook_new)

    # Write clustering and codebook to file
    output_dir = Path(args.input_file).parent / "results/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Write results")

    output_file = get_clustering_filename(
        args.input_file,
        args.embedding_model_name,
        args.distance_threshold,
        type_="codebook",
        suffix=".json",
    )
    output_file = Path(output_dir) / output_file
    json.dump(codebook, open(output_file, "w"))
    print(f".. Wrote codebook to: {output_file.absolute()}")

    output_file = get_clustering_filename(
        args.input_file,
        args.embedding_model_name,
        args.distance_threshold,
        type_="inverse_codebook",
        suffix=".json",
    )
    output_file = Path(output_dir) / output_file
    json.dump(inv_codebook, open(output_file, "w"))
    print(f".. Wrote inverse codebook to: {output_file.absolute()}")

    output_file = get_clustering_filename(
        args.input_file,
        args.embedding_model_name,
        args.distance_threshold,
        type_="clustering",
        suffix=".txt",
    )
    output_file = Path(output_dir) / output_file
    with open(output_file, "w") as fout:
        cluster_names = sorted(list(inv_codebook.keys()))
        for cluster_name in cluster_names:
            cluster = inv_codebook[cluster_name]
            if len(cluster) > 0:
                for ref in inv_codebook[cluster_name]:
                    fout.write(ref + "\n")
                fout.write("\n")
    print(f".. Wrote human-readable clustering to: {output_file.absolute()}")

    # Augment references
    references_augmented_column = []
    for references in data[REFERENCES_COLUMN]:
        references_augmented = "; ".join(
            [codebook[ref.strip()] for ref in references.split(";")]
        )
        references_augmented_column.append(references_augmented)
    data[REFERENCES_AUGMENTED_COLUMN] = references_augmented_column

    # Write augmented data to file
    output_file = get_clustering_filename(
        args.input_file,
        args.embedding_model_name,
        args.distance_threshold,
        type_="augmented",
        suffix=".csv",
    )
    output_file = Path(output_dir) / output_file
    data.to_csv(output_file, index=False)
    print(f".. Wrote augmented data to {output_file}")

    # Done
    print("Done.")


if __name__ == "__main__":
    main()
