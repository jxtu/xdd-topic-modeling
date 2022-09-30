from pathlib import Path
import re
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


if __name__ == '__main__':
    full_text_dir = "../xdd-covid-19-8Dec-doc2vec_text"
    count_all = 0
    count_has_abs = 0
    for p in tqdm.tqdm(Path(full_text_dir).iterdir()):
        # print(p)
        count_all += 1
        text = open(p, "r").read()
        abs_txt = extract_abs_from_txt(text)
        if abs_txt:
            count_has_abs += 1
        # print("====" * 10)
    print(count_all, count_has_abs, count_has_abs / count_all)
