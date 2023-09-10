# combine the bibjson and doc2vecs from the topic_doc2vecs/ folder
from pathlib import Path
import numpy as np
import json
from gensim.models import KeyedVectors

DATA_DIR = Path("topic_doc2vecs")
np_d2v_file_name = "model_default_streamed_docvecs.kv"

TOPIC1 = "biomedical"
TOPIC2 = "geoarchive"
TOPIC3 = "molecular_physics"


def combine_all_instances():
    all_json_docs = []
    all_doc2vecs = []
    for topic in (TOPIC1, TOPIC2, TOPIC3):
        json_obj = json.load(open(DATA_DIR.joinpath(topic).joinpath(f"{topic}_docids.bibjson"), "r", encoding="utf-8"))
        keyed_vecs = KeyedVectors.load(str(DATA_DIR.joinpath(topic).joinpath("doc2vec").joinpath(np_d2v_file_name)))

        assert len(json_obj) == len(keyed_vecs), f"size do not match! ({len(json_obj)}, {len(keyed_vecs)})"

        for obj in json_obj:
            obj["topic"] = topic
            all_json_docs.append(obj)
        for doc in json_obj:
            all_doc2vecs.append(keyed_vecs[doc["_gddid"]].reshape((1, -1)))
    all_doc2vecs = np.concatenate(all_doc2vecs, axis=0)
    with open(DATA_DIR.joinpath("combined_docids.bibjson"), "w", encoding="utf-8") as f:
        json.dump(all_json_docs, f, indent=4)
    np.save(str(DATA_DIR.joinpath("combined_doc2vec.wv.vectors.npy")), all_doc2vecs)

    print(f"finished dumping {len(all_json_docs)} instances!")


if __name__ == '__main__':
    combine_all_instances()