import pickle

species_names = set()

for i in range(21):
    with open('species_names%i.txt' % (i), 'rb') as f:
        my_set = pickle.load(f)
        species_names.update(my_set)

print(len(my_set))

with open('species_names.txt', 'wb') as f:
   pickle.dump(species_names, f)

