# script from Ian, developed on Gensim 3.x
# Data service application providing access to the 'doc2vec' operation and others from the Gensim model.
#
import json
import os, sys
import logging

from gensim.models import Doc2Vec, Word2Vec
from gensim.models.keyedvectors import Doc2VecKeyedVectors
from gensim.utils import simple_preprocess
from nltk import word_tokenize
from nltk.corpus import stopwords
import requests

stop_words = stopwords.words('english')

logging.basicConfig(format='%(levelname)s :: %(asctime)s :: %(message)s', level=logging.DEBUG)

if "DATA_DIR" in os.environ:
    data_dir = os.environ['DATA_DIR']
else:
    data_dir = "./data/"

stop_words = stopwords.words('english')

kv_path = f"{data_dir}/docvecs.kv"
model_path = f"{data_dir}/model_streamed_doc2vec"

models = {}

try:
    model = Doc2Vec.load(model_path)
    logging.info("Done")
except:
    logging.warning("Error: models not found.")
if not os.path.exists(kv_path):
    model.docvecs.save(kv_path)
try:
    logging.info(f"Loading keyed vectors from '{kv_path}'...")
    kv = Doc2VecKeyedVectors.load(kv_path)
except:
    logging.warning("Error: keyed vectors not found.")

def preprocess(line):
    line = word_tokenize(line)  # Split into words.
    line = [w.lower() for w in line]  # Lower the text.
    line = [w for w in line if not w in stop_words]  # Remove stopwords
    return line

def get_bibjsons(pdf_names):
    docids=','.join(pdf_names)
    resp = requests.get(f"https://xdd.wisc.edu/api/articles?docids={docids}")
    bibjson = {}
    if resp.status_code == 200:
        data = resp.json()
        if 'success' in data:
            for i in data['success']['data']:
                bibjson[i['_gddid']] = i
        else:
            logging.error(f'Unable to find success key: {data}')
            bibjson = None #{"Error" : "Could not retrieve article data"}
    else:
        bibjson = None #{"Error" : "Could not retrieve article data"}
    return bibjson

def similar(docid, contents=""):
    """Data service operation: execute a vector query on the specified word."""

    # full list of docids is tored in kv.doctags
    try:
        vec = kv[docid]
    except KeyError:
        print(f"Document {docid} not in vectorspace -- running a live infer.", flush=True)
        document = preprocess(' '.join(simple_preprocess(contents)))
        vec = model.infer_vector(document)

    n_responses=10
    similar = kv.most_similar(positive=[vec], topn=n_responses+1)
    similar = [(i[0], i[1]) for i in similar if i[0] != docid][:10]
    bibjsons = get_bibjsons([i[0] for i in similar])
    resp = [{"bibjson" : bibjsons[i[0]], "score" : i[1]} for i in similar]
    return resp


if __name__ == '__main__':
    print("---- Similar by docid ----")
    print(similar("5fb2df01d76fca4a3fb8888c"))
    print("---- Similar by inference ----")
    print(similar(None, "covid-19 rates continue to be stable."))

