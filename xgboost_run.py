import os
import pyterrier as pt
import pandas as pd
import xgboost as xgb
import re

from sklearn.feature_extraction.text import TfidfVectorizer,  ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity


pt.init()
ds = pt.get_dataset("irds:disks45/nocr/trec-robust-2004")

INDEX_DIR = r"C:\Users\paulj\OneDrive\Desktop\InfoRetrival\InformationRetrival\index"
os.makedirs(INDEX_DIR, exist_ok=True)

index_ref = pt.IndexRef.of(INDEX_DIR)


topics = ds.get_topics().rename(columns={"title": "query"})
topics["query"] = topics["query"].apply(lambda x: re.sub(r'[^A-Za-z0-9 ]+', ' ', x))
qrels = ds.get_qrels()

split = int(len(topics) * 0.8)
train_topics = topics.iloc[:split]
valid_topics = topics.iloc[split:]
train_qrels = qrels[qrels["qid"].isin(train_topics["qid"])]
valid_qrels = qrels[qrels["qid"].isin(valid_topics["qid"])]

# pipeline = pt.terrier.FeaturesRetriever(
#     index_ref,
#     wmodel="BM25",
#     features=["WMODEL:BM25"]
# )

bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25")
tf = pt.terrier.Retriever(index_ref, wmodel="Tf")
pl2 = pt.terrier.Retriever(index_ref, wmodel="PL2")

pipeline = bm25 >> (tf ** pl2)


xgb_ranker = xgb.XGBRanker(
    objective="rank:ndcg",
    learning_rate=0.1,
    gamma=1.0,
    min_child_weight=0.1,
    max_depth=6,
    random_state=42
)

# def keyword_overlap(row):
#     print(row)
#     query_terms = set(re.findall(r'\w+', row['query'].lower()))
#     doc_terms = set(re.findall(r'\w+', row['text'].lower()))
#     query_terms -= set(ENGLISH_STOP_WORDS)
#     doc_terms -= set(ENGLISH_STOP_WORDS)
#     return len(query_terms.intersection(doc_terms))

# overlap_query_title = pt.apply.doc_score(keyword_overlap)


ltr_pipeline = pipeline >> pt.ltr.apply_learned_model(xgb_ranker, form="ltr")

ltr_pipeline.fit(train_topics, train_qrels, valid_topics, valid_qrels)
results = ltr_pipeline.transform(valid_topics)
results.to_csv("ltr_results.csv", index=False)


print(results.head(10))

#trained_model = ltr_pipeline[-1].learner


evalution = pt.Experiment([ltr_pipeline],valid_topics,valid_qrels, eval_metrics=["map", "ndcg_cut_10"])
print(pd.DataFrame(evalution))

