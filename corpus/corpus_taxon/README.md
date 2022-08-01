`corpus_taxon.py` generates three main files:
1. `corpus_taxon.txt` (in `/corpus/corpus_taxon`): a text file that contains all of the non one-word taxon names without duplicates (approx. 870757)
2. `possible_species.pkl` (in `/ml-herbarium-data/corpus_taxon/output`): a dictionary of lists of possible species of each genus (key is genus and value is a list of possible species)
3. `possible_genus.pkl` (in `/ml-herbarium-data/corpus_taxon/output`): a dictionary of lists of possible genera of each species (key is species and value is a list of possible genera)

