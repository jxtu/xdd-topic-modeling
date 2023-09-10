from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
import tqdm
from pathlib import Path
import json
import numpy as np


def extract_title_abs(text_dict: Dict[str, Any]):
    abs_dict = text_dict.get("abstract")
    if abs_dict:
        abs_text = abs_dict.get("abstract")
        if abs_text and len(abs_text.strip().split()) > 1:
            abs = " ".join(abs_text.strip().split())
        else:
            abs = ""
    else:
        abs = ""

    title = text_dict.get("title")
    if title is None:
        title = ""
    return " ".join(title.strip().split()), abs


def extract_content(text_dict: Dict[str, Any], limit=1000):
    texts = []
    for sec_dict in text_dict["sections"]:
        text = sec_dict["text"]
        if text:
            text = text.strip().split()
            texts.extend(text)
    return " ".join(texts[:limit])


def extract_from_dir(text_dir, extract_type: str):
    gdd_ids = []
    texts = []
    for p in tqdm.tqdm(Path(text_dir).iterdir(), "Extracting text..."):
        if p.suffix == ".json":
            if extract_type == "title":
                gdd_id = p.stem
                full_text_dict = json.load(open(p, "r", encoding="utf-8"))
                title, abs = extract_title_abs(full_text_dict)
                gdd_ids.append(gdd_id)
                texts.append((title, abs))
            elif extract_type == "content":
                gdd_id = p.stem
                full_text_dict = json.load(open(p, "r", encoding="utf-8"))
                text = extract_content(full_text_dict)
                gdd_ids.append(gdd_id)
                texts.append(text)
            else:
                raise ValueError("unknown type")

    return gdd_ids, texts


def encode_text(step: int, texts: List[str]):
    all_embeddings = []
    model = SentenceTransformer('allenai-specter')
    for idx in tqdm.tqdm(range(0, len(texts), step), "Encoding text..."):
        query_embeddings = model.encode(texts[idx: idx + step])
        all_embeddings.append(query_embeddings)
    all_embeddings = np.concatenate(np.array(all_embeddings), axis=0)
    return all_embeddings


if __name__ == '__main__':
    topic = "molecular_physics"
    topic_dir = Path(__file__).parent.parent.joinpath(f"topic_doc2vecs/{topic}")
    text_dir = topic_dir.joinpath("processed")

    gdd_ids, texts = extract_from_dir(text_dir, extract_type="abstract")
    texts = [f"{title} [SEP] {abs}" for title, abs in texts]
    assert len(gdd_ids) == len(texts)
    out_f = open(topic_dir.joinpath(f"content_{topic}.txt"), "w", encoding="utf-8")
    for gid, text in zip(gdd_ids, texts):
        try:
            out_f.write("\t".join([gid, text]) + "\n")
        except UnicodeError:
            print("error", gid)
    out_f.close()
