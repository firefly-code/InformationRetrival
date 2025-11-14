import os
import pyterrier as pt
import pandas as pd
<<<<<<< HEAD
import ir_datasets
import os
=======
import xgboost as xgb
import re

from sklearn.feature_extraction.text import TfidfVectorizer,  ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

>>>>>>> 76ecae7d3202cbf31b98b3a0be8cee974dbbd570

pt.init()
ds = pt.get_dataset("irds:disks45/nocr/trec-robust-2004")

<<<<<<< HEAD
#need saved indexing method
ds = pt.get_dataset("irds:disks45/nocr/trec-robust-2004")
#dataset = ir_datasets.load(r"C:\Users\paulj\Documents\InformationRetrival\dataset\robust04\robust04")
#ds = pt.get_dataset(r"irds:C:\Users\paulj\Documents\InformationRetrival\dataset\robust04\robust04")

robust_path = r"C:\Users\paulj\Documents\InformationRetrival\dataset\robust04\robust04"

# indexer =  pt.TRECCollectionIndexer(r"C:\Users\paulj\Documents\InformationRetrival\InformationRetrival\indexdir")

# subfolders = [os.path.join(robust_path, f) for f in os.listdir(robust_path) 
#               if os.path.isdir(os.path.join(robust_path, f))]
# indexing = indexer.index(subfolders)
#pipeline = pt.apply.doc_features()
=======
INDEX_DIR = r"C:\Users\paulj\OneDrive\Desktop\InfoRetrival\index"
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
>>>>>>> 76ecae7d3202cbf31b98b3a0be8cee974dbbd570

topics= ds.get_topics()
qrels = ds.get_qrels()

<<<<<<< HEAD
shuffled = topics.sample(frac=1, random_state=42)
split = int(0.8 * len(shuffled))

train_topics = shuffled.iloc[:split]
validation_topics = shuffled.iloc[split:]

train_qrels = qrels
validation_qrels = qrels
#need access to data
#train_topics, train_qrels, validation_topics, validation_qrels = None


trec_files = []
for root, dirs, files in os.walk(robust_path):
    for f in files:
        trec_files.append(os.path.join(root, f))

print(f"Found {len(trec_files)} files to index.")

# Create Terrier index
indexer = pt.TRECCollectionIndexer(r"C:\Users\paulj\Documents\InformationRetrival\InformationRetrival\indexdir\rev_index_robust")
index_ref = indexer.index(trec_files)

#example feature
numf = 2
bm25 = pt.BatchRetrieve("irds:trec-robust04", wmodel="BM25")
pl2 = pt.BatchRetrieve("irds:trec-robust04", wmodel="PL2")

pipeline = bm25
pipeline = pipeline >> pl2

xg_model = xgb.XGBRanker(objective = "rank:MAP",
    learning_rate = 0.1,
    gamma=1.0,
    min_child_weight=0.1,
    max_depth=6,
    verbose=2,
    random_state=42 
)
rankers = []
xg_model_pipe = pipeline >> pt.ltr.apply_learned_model(xg_model, form="ltr")
xg_model_pipe.fit
rankers.append(xg_model_pipe(train_topics, train_qrels, validation_topics, validation_qrels))

# evaluate the full (4 features) model, as well as the each model containing only 3 features)
eval= pt.Experiment(
    rankers,
    validation_topics,
    validation_qrels,
    metrics=["map","recip_rank","P_5","R_5"],
    names=["Full Model"]  + ["Full Minus %d" % fid for fid in range(numf)]
=======
xgb_ranker = xgb.XGBRanker(
    objective="rank:ndcg",
    learning_rate=0.1,
    gamma=1.0,
    min_child_weight=0.1,
    max_depth=6,
    random_state=42
>>>>>>> 76ecae7d3202cbf31b98b3a0be8cee974dbbd570
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

trained_model = ltr_pipeline[-1].learner


evalution = pt.Experiment([ltr_pipeline],valid_topics,valid_qrels, eval_metrics=["map", "ndcg_cut_10"])
print(pd.DataFrame(evalution))

