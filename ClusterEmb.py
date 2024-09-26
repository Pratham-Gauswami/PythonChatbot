import pandas as pd
import numpy as np
import altair as alt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import cohere

co = cohere.Client("w9BCnpVENCLMBfaUkuSH1hEGrWNKexfD4N9aq3X3")  # Your Cohere API key

# Load the dataset to a dataframe
df_orig = pd.read_csv('https://raw.githubusercontent.com/cohere-ai/notebooks/main/notebooks/data/atis_intents_train.csv', names=['intent', 'query'])

# Take a small sample for illustration purposes
sample_classes = ['atis_airfare', 'atis_airline', 'atis_ground_service']
df = df_orig.sample(frac=0.1, random_state=30)
df = df[df.intent.isin(sample_classes)]
df_orig = df_orig.drop(df.index)
df.reset_index(drop=True, inplace=True)

# Remove unnecessary column
intents = df['intent']  # save for a later need
df.drop(columns=['intent'], inplace=True)

def get_embeddings(texts, model='embed-english-v3.0', input_type="clustering"):
    output = co.embed(
        model=model,
        input_type=input_type,
        texts=texts)
    return output.embeddings

# Embed the text for clustering
df['clustering_embeds'] = get_embeddings(df['query'].tolist(), input_type="clustering")
embeds = np.array(df['clustering_embeds'].tolist())

# Pick the number of clusters
n_clusters = 2

# Cluster the embeddings
kmeans_model = KMeans(n_clusters=n_clusters, random_state=0)
classes = kmeans_model.fit_predict(embeds).tolist()

# Store the cluster assignments in the original DataFrame
df['cluster'] = classes

# Print the first few rows of the DataFrame with clusters
print(df.head())

# Function to return the principal components
def get_pc(arr, n):
    pca = PCA(n_components=n)
    embeds_transform = pca.fit_transform(arr)
    return embeds_transform

# Reduce embeddings to 10 principal components to aid visualization
df['query_embeds'] = get_embeddings(df['query'].tolist())  # Get the embeddings for queries
embeds = np.array(df['query_embeds'].tolist())
embeds_pc = get_pc(embeds, 10)

# Define new query
new_query = "How can I find a taxi or a bus when the plane lands?"

# Get embeddings of the new query
new_query_embeds = get_embeddings([new_query], input_type="search_query")[0]

# Calculate cosine similarity between the search query and existing queries
def get_similarity(target, candidates):
    # Turn list into array
    candidates = np.array(candidates)
    target = np.expand_dims(np.array(target), axis=0)

    # Calculate cosine similarity
    sim = cosine_similarity(target, candidates)
    sim = np.squeeze(sim).tolist()
    sort_index = np.argsort(sim)[::-1]
    sort_score = [sim[i] for i in sort_index]
    similarity_scores = zip(sort_index, sort_score)

    # Return similarity scores
    return similarity_scores

# Get the similarity between the search query and existing queries
similarity = get_similarity(new_query_embeds, embeds)

# View the top 5 articles
print('Query:')
print(new_query, '\n')

print('Most Similar Documents:')
for idx, sim in similarity:
    print(f'Similarity: {sim:.2f};', df.iloc[idx]['query'])

# Visualization using Altair
pca_df = pd.DataFrame(embeds_pc, columns=[f'PC{i+1}' for i in range(embeds_pc.shape[1])])
pca_df['cluster'] = df['cluster']

chart = alt.Chart(pca_df).mark_circle(size=60).encode(
    x='PC1',
    y='PC2',
    color='cluster:N',
    tooltip=['PC1', 'PC2', 'cluster']
).interactive()

chart.display()
