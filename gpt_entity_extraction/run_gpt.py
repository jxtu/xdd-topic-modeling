import json
import os
import openai
import tqdm

openai.api_key = os.getenv("OPENAI_API_KEY")

COMPLETION_PARAMS = {
    # TODO: any other parameters we need to change?
    "model": "gpt-4",
    "temperature": 0,
    "max_tokens": 200,  # max tokens in the output
    "messages": [],
}

with open("sample_text_620cf21cad0e9c819b05041a.txt", "r") as f:
    sample_texts = f.readlines()


def generate_prompts(text: str, option: int):
    prompt_mapping = {
        0: "What are the geological entities from the following text?",
        1: "What are the geological entities and their types from the following text? Entity type can only be one word.",
        2: "Extract the geological entities and relations from the following text. Each relation tuple should be formatted as (entity1, event, entity2)."
    }

    prompt = prompt_mapping[option]
    return "\n\n".join([prompt, text])


def generate_message(prompt_str: str):
    # TODO: should we try other roles or generate the message in the "chat completion" style
    return [{"role": "user", "content": prompt_str}]


def write_gpt_res_file():
    option = 1

    out_f = open(f"gpt_output_{option}.jsonl", "w")

    for text in tqdm.tqdm(sample_texts, "running GPT"):
        prompted_input = generate_prompts(text, option)
        COMPLETION_PARAMS["messages"] = generate_message(prompted_input)
        response = openai.ChatCompletion.create(**COMPLETION_PARAMS)
        content = response.choices[0].message.content
        out_f.write(json.dumps({"instruction": prompted_input, "response": content}) + "\n")
    out_f.close()


if __name__ == '__main__':
    write_gpt_res_file()
