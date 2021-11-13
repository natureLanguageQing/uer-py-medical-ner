import json

medical_ner = json.load(open("medical_ner.json","r"))

print(medical_ner)
labels =[]
for medical_ner_one in medical_ner:
    labels.extend(medical_ner_one["ner_tags"])
labels = list(set(labels))
out_label_id = {}
for index, label in enumerate(labels):
    out_label_id[label] = index
json.dump(out_label_id,open("medical_label2id.json","w"),ensure_ascii=False)