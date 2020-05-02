#!/usr/bin/env bash

# Download Genre-Balanced American English embeddings with word lemmas (2.9GB)
# Multiple historical embedding types+detailed historical statistics files from 
# https://nlp.stanford.edu/projects/histwords/ 
# Generated from COHA datasets https://www.english-corpora.org/coha/ 
if [ ! -f ./coha-lemma.zip ]; then
	echo "Downloading COHA embeddings w word lemmas (2.9GB)..."
	curl -Lo ./coha-lemma.zip http://snap.stanford.edu/historical_embeddings/coha-lemma.zip
	unzip ./coha-lemma.zip
fi

# w/o lemmas (4.6GB) available at http://snap.stanford.edu/historical_embeddings/coha-word.zip
