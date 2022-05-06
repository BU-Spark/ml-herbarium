import pickle 
with open('syn_pure_vscode.pkl', 'rb') as f:
    syn_modi = pickle.load(f)

print(syn_modi['Hieracium murorum'])