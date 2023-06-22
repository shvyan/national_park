import csv
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Read movie data from the CSV file
National_Parks = []
with open('/content/National_Parks.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        National_Parks.append(row)

# Encode the descriptions and generate BERT embeddings
embeddings = []
for row in National_Parks:
    description = row['DESCRIPTION']
    # Tokenize the description
    tokens = tokenizer.encode(description, add_special_tokens=True)
    # Convert tokens to tensors
    input_ids = torch.tensor(tokens).unsqueeze(0)
    # Generate BERT embeddings
    with torch.no_grad():
        outputs = model(input_ids)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())

# Reshape the embeddings
embeddings = torch.tensor(embeddings)

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(embeddings)

# Now, suppose a user likes "Movie 1". We can recommend another movie based on cosine similarity.
liked_national_parks = "Serengeti National Park"
liked_national_parks_index = next(index for index, national_park in enumerate(National_Parks) if national_park['NATIONAL_PARKS'] == liked_national_parks)

# Find the most similar movie
similar_national_park_index = similarity_matrix[liked_national_parks_index].argsort()[::-1][1]  # Exclude the liked movie itself
recommended_national_parks = National_Parks[similar_national_park_index]

print("Because you liked " + liked_national_parks + ", we recommend: " + recommended_national_parks['NATIONAL_PARKS'])
