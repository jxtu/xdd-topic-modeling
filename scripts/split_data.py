import json
import numpy as np
import click
from pathlib import Path

DATA_DIR = Path("xdd-covid-19-8Dec-doc2vec")
json_file_name = "xdd-covid-19-8Dec.bibjson"
np_d2v_file_name = "model_streamed_doc2vec.docvecs.vectors_docs.npy"
SAMPLE_DATA_DIR = Path("sample_data")

SAMPLE_SIZE = 100


def sample_top_k(sample_size, in_json_file, in_np_file) -> None:
    json_obj = json.load(open(DATA_DIR.joinpath(in_json_file), "r", encoding="utf-8"))

    np_obj = np.load(str(DATA_DIR.joinpath(in_np_file)))
    assert len(json_obj) == np_obj.shape[0], "size do not match!"
    # NOTE: assuming the document order is already random, we take the top k for sampling
    sampled_json_obj = json_obj[:sample_size]
    sampled_np_obj = np_obj[:sample_size]
    with open(DATA_DIR.joinpath(f"top_{sample_size}_{in_json_file}"), "w", encoding="utf-8") as f:
        json.dump(sampled_json_obj, f, indent=4)
    np.save(str(DATA_DIR.joinpath(f"top_{sample_size}_{in_np_file}")), sampled_np_obj)

    print(f"finished dumping top {sample_size} instances!")


@click.command()
@click.argument("sample_size", type=click.INT, default=SAMPLE_SIZE)
def main(sample_size: int):
    sample_top_k(sample_size, json_file_name, np_d2v_file_name)


if __name__ == '__main__':
    # python -m scripts.split_data 200
    main()
