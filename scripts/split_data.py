import json
import numpy as np

from pathlib import Path

DATA_DIR = Path("xdd-covid-19-8Dec-doc2vec")
json_file_name = "xdd-covid-19-8Dec.bibjson"
np_d2v_file_name = "model_streamed_doc2vec.docvecs.vectors_docs.npy"
SAMPLE_DATA_DIR = Path("sample_data")

SAMPLE_SIZE = 100


def sample_top_k(in_json_file, in_np_file) -> None:
    json_obj = json.load(open(DATA_DIR.joinpath(in_json_file), "r", encoding="utf-8"))

    np_obj = np.load(str(DATA_DIR.joinpath(in_np_file)))
    assert len(json_obj) == np_obj.shape[0], "size do not match!"
    sampled_json_obj = json_obj[:SAMPLE_SIZE]
    sampled_np_obj = np_obj[:SAMPLE_SIZE]
    with open(SAMPLE_DATA_DIR.joinpath(f"top_{SAMPLE_SIZE}_{in_json_file}"), "w", encoding="utf-8") as f:
        json.dump(sampled_json_obj, f, indent=4)
    np.save(str(SAMPLE_DATA_DIR.joinpath(f"top_{SAMPLE_SIZE}_{in_np_file}")), sampled_np_obj)

    print(f"finished dumping top {SAMPLE_SIZE} instances!")


if __name__ == '__main__':
    # python -m scripts.split_data
    sample_top_k(json_file_name, np_d2v_file_name)
