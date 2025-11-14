import pyterrier as pt 
import xgboost as xgb
import pandas as pd
import ir_datasets
import os

pt.init()

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

topics= ds.get_topics()
qrels = ds.get_qrels()

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
)

eval_results = eval.run()

print(pd.DataFrame(eval_results))
