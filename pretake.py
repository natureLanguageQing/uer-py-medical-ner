import os

medical_all_data = open("corpora/medical_all_data.txt", "r").readlines()
nest = []
for i in medical_all_data:
    i = i.strip("\n")
    if "\t" in i:
        nest.extend(i.split("\t"))
    else:
        nest.append(i)
nest = list(set(nest))
open("corpora/medical_data.txt", "w").write("\n".join(nest))
