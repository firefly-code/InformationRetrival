import pyterrier as pt 
import xgboost as xgb
import pandas as pd




pt.init()

#need saved indexing method
indexing = pt.IndexRef.of("rev_indexing_path")
#pipeline = pt.apply.doc_features()


#need access to data
train_topics, train_qrels, validation_topics, validation_qrels = None


#example feature
numf = 2
bm25 = pt.BatchRetrieve(indexing, wmodel="BM25")
pl2 = pt.BatchRetrieve(indexing, wmodel="PL2")

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
xg_model_pipe.fit(train_topics, train_qrels, validation_topics, validation_qrels)
rankers.append(xg_model_pipe)

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
