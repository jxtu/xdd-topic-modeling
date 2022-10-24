import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from contextualized_topic_models.models.ctm import ZeroShotTM, CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessingStopwords
from nltk.corpus import stopwords as stop_words

stopwords = set(stop_words.words("english"))


def preprocess_docs(json_file_path):
    json_abs = json.load(open(json_file_path, "r", encoding="utf-8"))
    docs_ggids, docs = zip(*json_abs.items())
    print(len(docs))
    sp = WhiteSpacePreprocessingStopwords(docs[:100], stopwords_list=stopwords)
    prep_docs, unprep_doc, vocab, retained_idx = sp.preprocess()
    return docs_ggids, prep_docs, unprep_doc


if __name__ == '__main__':
    docs_ggids, prep_docs, unprep_doc = preprocess_docs("../xdd-covid-19-8Dec-doc2vec/xdd-covid-19-abs-sci–parser.json")
    qt = TopicModelDataPreparation("allenai-specter")
    training_dataset = qt.fit(text_for_contextual=prep_docs, text_for_bow=unprep_doc)

    ctm = ZeroShotTM(bow_size=len(qt.vocab), contextual_size=768, n_components=10, batch_size=32, num_epochs=20)

    ctm.fit(training_dataset)  # run the model

    topics = ctm.get_topics(5)

    print(topics)
