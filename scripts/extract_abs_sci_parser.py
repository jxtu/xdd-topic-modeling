import json
from pathlib import Path
from typing import Dict, Any

import click
import tqdm


def extract_abs_from_sci_parser_output(text_dict: Dict[str, Any]):
    abs_text = text_dict["metadata"].get("abstractText", "")
    if abs_text and len(abs_text.strip().split()) > 1:
        return abs_text.strip()
    else:
        return ""


@click.command()
@click.argument("text_dir", type=click.Path(exists=True), default="xdd-covid-19-8Dec-scienceparse/scienceparse")
@click.argument("out_json_file", type=click.Path(), default="xdd-covid-19-abs-sci–parser.json")
def abs_to_json(text_dir, out_json_file) -> None:
    count_all = 0
    count_has_abs = 0
    abs_dict = dict()
    for p in tqdm.tqdm(Path(text_dir).iterdir()):
        if p.suffix == ".json":
            gdd_id = p.stem.split("_")[0]
            count_all += 1
            full_text_dict = json.load(open(p, "r", encoding="utf-8"))
            abs_txt = extract_abs_from_sci_parser_output(full_text_dict)
            if abs_txt:
                abs_dict[gdd_id] = abs_txt
                count_has_abs += 1
    with open(out_json_file, "w") as f:
        f.write(json.dumps(abs_dict, indent=4))
    print(f"extracted {count_has_abs} abstracts from {count_all} articles")


if __name__ == '__main__':
    abs_to_json()
