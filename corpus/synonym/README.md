`synonym.py` generates three pickle files:

1. dic.pkl: It is a dictionary of taxonID and its scientific name. Key is taxonID and its value is [i, scientific name] where i is the index.
2. syn.pkl: It is a dictionary of pairs of synonyms. Key is the synonym itself and its value is the acceptedName which is the most updated name.
3. syn_pure.pkl: It is basically syn.pkl but without the collector names and the year.
