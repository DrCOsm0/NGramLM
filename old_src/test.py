import pickle

# Your Python list
my_list = [1, 2, 3, 4, 5]

# Save the list to a file
with open('output.pkl', 'wb') as file:
    pickle.dump(my_list, file)

# Load the list back from the file
with open('output.pkl', 'rb') as file:
    loaded_list = pickle.load(file)

# Print the loaded list
print(loaded_list)