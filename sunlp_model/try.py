from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import json
import ipdb

pretrained_model_name = "busecarik/bert-loodos-sunlp-ner-turkish"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
model = AutoModelForTokenClassification.from_pretrained(pretrained_model_name)
ner_pipeline = pipeline('ner', model=model, tokenizer=tokenizer)

import pymongo
# Connect to mongodb
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["politus_twitter"]
tweet_col = db["tweets"]

query = {"text": {"$nin": ["", None]}}
with open("out.json", "w", encoding="utf-8") as f:
    for idx, row in enumerate(tweet_col.find(query, ["_id", "text"])):
        if idx == 100: break
        res = ner_pipeline(row["text"])
        for r in res:
            for k,v in r.items():
                r[k] = str(v)
        print(res)
        out_d = {"id": row["_id"], "text": row["text"], "preds": res}
        f.write(json.dumps(out_d, ensure_ascii=False) + "\n")
