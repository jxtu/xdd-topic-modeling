import json
import pprint
from pathlib import Path
from typing import List

import requests
import tqdm


def get_meta_from_api(doc_ids: List[str]):
    api_url = f"https://xdd.wisc.edu/api/articles?docids={','.join(doc_ids)}"
    response = requests.get(api_url)
    json_obj = response.json()
    data_lst = json_obj["success"]["data"]
    return data_lst


def write_meta_to_file(topic: str):
    topic_data_path = Path(__file__).parent.parent.joinpath("topic_doc2vecs").joinpath(topic)
    processed_data_path = topic_data_path.joinpath("processed")
    out_file_path = topic_data_path.joinpath("meta.jsonl")
    out_f = open(out_file_path, "w")
    doc_ids = [p.stem for p in processed_data_path.iterdir()]
    step = 100
    for i in tqdm.tqdm(range(0, len(doc_ids), step)):
        data_lst = get_meta_from_api(doc_ids[i: i + step])
        for d in data_lst:
            out_f.write(json.dumps(d) + "\n")
    out_f.close()


def parse_meta_jsonl(meta_jsonl_file: str, out_meta_jsonl_file: str):
    # ['type', '_gddid', 'title', 'volume', 'journal', 'link', 'publisher', 'abstract',
    # 'author', 'pages', 'number', 'identifier', 'year'])

    keys = "_gddid title journal publisher author".split()
    out_f = open(out_meta_jsonl_file, "w")

    with open(meta_jsonl_file, "r") as f:
        for line in f:
            meta_dict = dict()
            obj = json.loads(line)
            for k in keys:
                meta_dict[k] = obj.get(k)
            print(meta_dict)
            out_f.write(json.dumps(meta_dict) + "\n")


if __name__ == '__main__':
    topic = "molecular_physics"
    # write_meta_to_file(topic)

    topic_data_path = Path(__file__).parent.parent.joinpath("topic_doc2vecs").joinpath(topic)
    meta_file_path = str(topic_data_path.joinpath("meta.jsonl"))
    filtered_meta_file_path = str(topic_data_path.joinpath("filtered_meta.jsonl"))

    parse_meta_jsonl(meta_file_path, filtered_meta_file_path)
