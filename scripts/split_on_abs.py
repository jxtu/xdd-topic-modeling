import json
import numpy as np
from pathlib import Path

DATA_DIR = Path("xdd-covid-19-8Dec-doc2vec")
json_file_name = "xdd-covid-19-8Dec.bibjson"
np_d2v_file_name = "model_streamed_doc2vec.docvecs.vectors_docs.npy"
abs_json_file = "xdd-covid-19-abs-sci–parser.json"  # or "xdd-covid-19-abs-regex.json"


def sample_article_with_abs(sample_size, in_json_file, in_np_file, in_abs_file) -> None:
    json_objs = json.load(open(DATA_DIR.joinpath(in_json_file), "r", encoding="utf-8"))
    json_abs = json.load(open(DATA_DIR.joinpath(in_abs_file), "r", encoding="utf-8"))
    np_objs = np.load(str(DATA_DIR.joinpath(in_np_file)))

    sampled_json_objs = []
    sampled_np_objs = []

    assert len(json_objs) == np_objs.shape[0], "size do not match!"
    for i, json_obj in enumerate(json_objs):
        ggd_id = json_obj["_gddid"]
        if json_abs.get(ggd_id):
            json_obj["abs"] = json_abs.get(ggd_id)
            sampled_json_objs.append(json_obj)
            sampled_np_objs.append(np_objs[i])

    with open(DATA_DIR.joinpath(f"top_{sample_size}_abs_{in_json_file}"), "w", encoding="utf-8") as f:
        json.dump(sampled_json_objs[:sample_size], f, indent=4)
    np.save(str(DATA_DIR.joinpath(f"top_{sample_size}_abs_{in_np_file}")), np.array(sampled_np_objs[:sample_size]))

    print(f"finished dumping instances with abstract!")


if __name__ == '__main__':
    sample_size = 50000
    sample_article_with_abs(sample_size, json_file_name, np_d2v_file_name, abs_json_file)

