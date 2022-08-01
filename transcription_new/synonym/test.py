import pickle 
with open('/projectnb/sparkgrp/ml-herbarium-grp/ml-herbarium-data/synonym-matching/output/syn_pure.pkl', 'rb') as f:
    syn_dict = pickle.load(f)

test = "Hypericum formosum"

print("Input: " + test)

if (test.lower() in syn_dict):
    test = syn_dict[test.lower()]
    print("Synonym: " + test)
