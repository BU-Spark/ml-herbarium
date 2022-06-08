import pickle 
with open('syn_pure.pkl', 'rb') as f:
    syn_modi = pickle.load(f)

print(syn_modi['Carex echinata'])
# Carex muricata