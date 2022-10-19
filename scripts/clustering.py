import json
from collections import Counter, defaultdict
from typing import List
from math import log2

import click
import numpy as np
from sklearn.cluster import KMeans
from nltk.corpus import stopwords

from scripts.cluster_viz import scatter_viz

SEED = 42
DEFAULT_TITLE = "NONE"

CUSTOMIZED_STOPWORDS = ("covid-19", "coronavirus", "sars-cov-2", "virus", "patients", "pandemic",
                        "infection", "disease", "study", "protein", "analysis")
STOPWORDS = set(stopwords.words('english'))
STOPWORDS.update(CUSTOMIZED_STOPWORDS)


def load_d2v(npy_file) -> "np.ndarray":
    d2v_array = np.load(npy_file)
    return d2v_array


def kmeans_cluster(d2v_array: "np.ndarray", n_cluster: int) -> "np.ndarray":
    kmeans = KMeans(n_clusters=n_cluster, random_state=SEED)
    kmeans.fit(d2v_array)
    return kmeans.labels_


def select_keyword_by_freq(title_list: List[str], top_k: int):
    word_counter = Counter()
    for title in title_list:
        title_words = [tok for tok in title.strip().split() if tok.lower() not in STOPWORDS and len(tok) > 2]
        word_counter.update(title_words)
    return [w[0] for w in word_counter.most_common(top_k)]


def idf(N: int, df: int) -> float:
    """compute the idf score, given df and document count N"""
    return log2(N / df)


def tf(freq: int) -> float:
    """compute tf score, given the term count freq in a document"""
    return 1.0 + log2(freq)


def select_keyword_by_tfidf(doc_list: List[str], top_k: int):
    index_dict = defaultdict(int)
    N = len(doc_list)
    for doc in doc_list:
        words = set(tok for tok in doc.strip().split() if tok.lower() not in STOPWORDS and len(tok) > 2)
        for word in words:
            index_dict[word] += 1

    word_counter = Counter()

    for doc in doc_list:
        words = [tok for tok in doc.strip().split() if tok.lower() not in STOPWORDS and len(tok) > 2]
        terms_counter = Counter(words)
        term_weights = {}
        for term in terms_counter:
            term_tf = tf(terms_counter[term])
            term_idf = idf(N, index_dict[term])
            term_weights[term] = term_tf * term_idf
        sorted_term_weights = sorted(term_weights.items(), key=lambda k: k[1], reverse=True)
        topk_words = [k for k, v in sorted_term_weights[:top_k]]
        word_counter.update(topk_words)
    return [w[0] for w in word_counter.most_common(top_k)]


def get_cluster_keywords(cluster_labels: "np.ndarray", tok_k: int, in_json_file, field: str):
    json_obj = json.load(open(in_json_file, "r", encoding="utf-8"))
    doc_id_cluster_dict = defaultdict(list)
    for doc_id, cluster_label in enumerate(cluster_labels):
        doc_id_cluster_dict[cluster_label].append(doc_id)

    doc_title_cluster_dict = defaultdict(list)
    for cluster_id in doc_id_cluster_dict:
        for doc_id in doc_id_cluster_dict[cluster_id]:
            doc_title_cluster_dict[cluster_id].append(json_obj[doc_id].get(field, DEFAULT_TITLE))
    print("Cluster", "Count", "Keywords", sep="\t")
    for cluster_id in doc_title_cluster_dict:
        print(cluster_id, len(doc_title_cluster_dict[cluster_id]),
              select_keyword_by_tfidf(doc_title_cluster_dict[cluster_id], tok_k), sep="\t")


@click.command()
@click.argument("file_prefix", type=click.STRING)
@click.argument("n_cluster", type=click.INT)
@click.argument("top_k", type=click.INT)
@click.option('--field', type=click.Choice(['abs', 'title'], case_sensitive=True))
def main(file_prefix: int, n_cluster: int, top_k: int, field):
    npy_file_path = f"xdd-covid-19-8Dec-doc2vec/{file_prefix}_model_streamed_doc2vec.docvecs.vectors_docs.npy"
    json_file_path = f"xdd-covid-19-8Dec-doc2vec/{file_prefix}_xdd-covid-19-8Dec.bibjson"
    array = load_d2v(npy_file_path)
    labels = kmeans_cluster(array, n_cluster)
    scatter_viz(array, labels, f"scatter_{field}_{file_prefix}", use_tne=False)
    get_cluster_keywords(labels, top_k, json_file_path, field)


if __name__ == '__main__':
    # python -m scripts.clustering top_1000 5 5 --field title
    main()
