import os

from representations.embedding import SOFTCOS_SPEARMAN_SIMILARITY, SOFTCOS_PEARSON_SIMILARITY
from representations.sequentialembedding import SequentialEmbedding

"""
Example showing how to load a series of historical embeddings and compute similarities over time.
Warning that loading all the embeddings into main memory can take a lot of RAM
"""

if __name__ == "__main__":
    embeddings_dir = os.path.abspath("../rsrc/coha-lemma/sgns")
    years = list(range(1950, 2000 + 1, 10))

    fiction_embeddings = SequentialEmbedding.load(embeddings_dir, years)

    print("Computing synchronic similarities (cosine).")
    time_sims = fiction_embeddings.get_time_sims("happy", "gay")
    print("Similarity between gay and happy from 1950s to the 2000s (cosine):")
    for year, sim in time_sims.items():
        print("{year:d}, cosine similarity = {sim:0.2f}".format(year=year, sim=sim))

    print("Computing synchronic similarities (softcos similarity with spearman).")
    time_sims = fiction_embeddings.get_time_sims("happy", "gay", similarity_measure=SOFTCOS_SPEARMAN_SIMILARITY)
    print("Similarity between gay and happy from 1950s to the 2000s (softcos similarity with spearman):")
    for year, sim in time_sims.items():
        print("{year:d}, softcos similarity with spearman = {sim:0.2f}".format(year=year, sim=sim))

    print("Computing synchronic similarities (softcos similarity with pearson).")
    time_sims = fiction_embeddings.get_time_sims("happy", "gay", similarity_measure=SOFTCOS_PEARSON_SIMILARITY)
    print("Similarity between gay and happy from 1950s to the 2000s (softcos similarity with pearson):")
    for year, sim in time_sims.items():
        print("{year:d}, softcos similarity with pearson = {sim:0.2f}".format(year=year, sim=sim))
