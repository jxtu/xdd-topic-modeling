import json
from collections import Counter, defaultdict

if __name__ == '__main__':
    k = 5
    with open(f"topic_doc2vecs/geoarchive/geoarchive_content_{k}_cluster_example.json", "r") as f:
        cluster_dict = json.load(f)

    topic = "geo"
    counter = Counter()
    with open(f"topic_doc2vecs/corpora/{topic}/tf.json", "r") as f:
        tf_dict = json.load(f)

    for c_id in cluster_dict:
        cluster_tf_dict = defaultdict(int)
        doc_ids = cluster_dict[c_id]
        for doc_id in doc_ids:
            tfs = tf_dict[f"{doc_id}.txt"]
            for tf in tfs:
                cluster_tf_dict[tf] += tfs[tf]
        sorted_dict = sorted(cluster_tf_dict.items(), key=lambda k: k[1], reverse=True)[:20]
        print(sorted_dict)