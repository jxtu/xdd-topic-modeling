import json
from pathlib import Path
import re

import click
import tqdm


def extract_abs_from_txt(text: str):
    text_lines = text.strip().split("\n")
    for i in range(len(text_lines) - 1, 0, -1):
        if re.match(r"introduction|keywords", text_lines[i].lower()):
            sub_lines = [l for l in text_lines[:i] if l.strip()]

            for j, line in enumerate(sub_lines):
                if re.match(r"abstract", line.lower()):
                    abs_text = " ".join(sub_lines[j:])
                    return abs_text
    return ""


@click.command()
@click.argument("text_dir", type=click.Path(exists=True), default="xdd-covid-19-8Dec-doc2vec_text")
@click.argument("out_json_file", type=click.Path(), default="xdd-covid-19-abs.json")
def abs_to_json(text_dir, out_json_file) -> None:
    count_all = 0
    count_has_abs = 0
    abs_dict = dict()
    for p in tqdm.tqdm(Path(text_dir).iterdir()):
        if p.suffix == ".txt":
            gdd_id = p.stem
            count_all += 1
            full_text = open(p, "r").read()
            abs_txt = extract_abs_from_txt(full_text)
            if abs_txt:
                abs_dict[gdd_id] = abs_txt
                count_has_abs += 1
    with open(out_json_file, "w") as f:
        f.write(json.dumps(abs_dict, indent=4))
    print(f"extracted {count_has_abs} abstracts from {count_all} articles")


if __name__ == '__main__':
    abs_to_json()
