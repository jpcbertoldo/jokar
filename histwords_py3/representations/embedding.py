"""
[1] https://en.wikipedia.org/wiki/Cosine_similarity#Soft_cosine_measure
"""

import heapq
import itertools
from typing import Optional

import numpy as np
import scipy as sp
from sklearn import preprocessing

from ioutils import load_pickle


COSINE_SIMILARITY = "cosine"
SOFTCOS_SPEARMAN_SIMILARITY = "softcos-spearman"
SOFTCOS_PEARSON_SIMILARITY = "softcos-pearson"
VALID_SIMILARITY_MEASURES = (COSINE_SIMILARITY, SOFTCOS_SPEARMAN_SIMILARITY, SOFTCOS_PEARSON_SIMILARITY)


def compute_similarity_matrix(emb, similarity_measure):
    # todo(joao) manage files to keep these things pre computed

    if similarity_measure == SOFTCOS_SPEARMAN_SIMILARITY:
        return np.abs(sp.stats.spearmanr(emb).correlation)

    elif similarity_measure == SOFTCOS_PEARSON_SIMILARITY:
        dim = emb.shape[1]
        correlation_matrix_pearson = np.zeros((dim, dim))
        for i, j in itertools.product(range(dim), range(dim)):
            if i <= j:
                pearson_ij = sp.stats.pearsonr(emb[i], emb[j])[0]
            else:
                pearson_ij = correlation_matrix_pearson[j, i]

            correlation_matrix_pearson[i, j] = pearson_ij
        return np.abs(correlation_matrix_pearson)


def _softcos(v1, v2, similarity_matrix):
    dim = len(similarity_matrix)
    a, b, ab = 0., 0., 0.
    for i, j in itertools.product(range(dim), range(dim)):
        ab += similarity_matrix[i, j] * v1[i] * v2[j]
        a += similarity_matrix[i, j] * v1[i] * v1[j]
        b += similarity_matrix[i, j] * v2[i] * v2[j]
    return ab / np.sqrt(a * b)


class Embedding:
    """
    Base class for all embeddings. SGNS can be directly instantiated with it.
    """

    def __init__(self, vecs, vocab, normalize=True, **kwargs):
        """
        Args:
            vecs:
            vocab:
            normalize:
            similarity_measure: if not None, the similarity will used will be soft cosine (see [1]),
                     the value tells which similarity is considered:
                        - softcos-spearman
                        - softcos-pearson
            **kwargs:
        """
        self.m = vecs
        self.dim = self.m.shape[1]
        self.iw = vocab
        self.wi = {w:i for i,w in enumerate(self.iw)}
        # todo(joao) do not normalize everything(?) or change this process to consider the covariance matrix
        if normalize:
            self.normalize()
        self.correlation_matrix_spearman = None
        self.correlation_matrix_pearson = None

    def __getitem__(self, key):
        if self.oov(key):
            raise KeyError
        else:
            return self.represent(key)

    def __iter__(self):
        return self.iw.__iter__()

    def __contains__(self, key):
        return not self.oov(key)

    @classmethod
    def load(cls, path, normalize=True, add_context=False, **kwargs):
        print(f"Loading Embedding from {path} normalize={normalize}")
        mat = np.load(path + "-w.npy", mmap_mode="c")
        if add_context:
            mat += np.load(path + "-c.npy", mmap_mode="c")
        iw = load_pickle(path + "-vocab.pkl")
        return cls(mat, iw, normalize) 

    def get_subembed(self, word_list, **kwargs):
        word_list = [word for word in word_list if not self.oov(word)]
        keep_indices = [self.wi[word] for word in word_list]
        return Embedding(self.m[keep_indices, :], word_list, normalize=False)

    def reindex(self, word_list, **kwargs):
        new_mat = np.empty((len(word_list), self.m.shape[1]))
        valid_words = set(self.iw)
        for i, word in enumerate(word_list):
            if word in valid_words:
                new_mat[i, :] = self.represent(word)
            else:
                new_mat[i, :] = 0 
        return Embedding(new_mat, word_list, normalize=False)

    def get_neighbourhood_embed(self, w, n=1000):
        neighbours = self.closest(w, n=n)
        keep_indices = [self.wi[neighbour] for _, neighbour in neighbours] 
        new_mat = self.m[keep_indices, :]
        return Embedding(new_mat, [neighbour for _, neighbour in neighbours]) 

    def normalize(self):
        preprocessing.normalize(self.m, copy=False)

    def oov(self, w):
        return not (w in self.wi)

    def represent(self, w: str) -> np.ndarray:
        if w in self.wi:
            return self.m[self.wi[w], :]
        else:
            print("OOV: ", w)
            return np.zeros(self.dim)

    def similarity(self, w1: str, w2: str, similarity_measure: Optional[str] = None) -> float:
        """
        Assumes the vectors have been normalized.
        """
        assert similarity_measure is None or similarity_measure in VALID_SIMILARITY_MEASURES, \
            f"Invalid similarity measure. Chose one of {VALID_SIMILARITY_MEASURES} (default is {COSINE_SIMILARITY})."

        w1_emb = self.represent(w1)
        w2_emb = self.represent(w2)

        if not w1_emb.any() or not w2_emb.any():
            return 0.

        if similarity_measure is None or similarity_measure == COSINE_SIMILARITY:
            return 1. - sp.spatial.distance.cosine(w1_emb, w2_emb)

        elif similarity_measure == SOFTCOS_SPEARMAN_SIMILARITY:
            if self.correlation_matrix_spearman is None:
                self.correlation_matrix_spearman = compute_similarity_matrix(self.m, SOFTCOS_SPEARMAN_SIMILARITY)
            return _softcos(w1_emb, w2_emb, self.correlation_matrix_spearman)

        elif similarity_measure == SOFTCOS_PEARSON_SIMILARITY:
            if self.correlation_matrix_pearson is None:
                self.correlation_matrix_pearson = compute_similarity_matrix(self.m, SOFTCOS_PEARSON_SIMILARITY)
            return _softcos(w1_emb, w2_emb, self.correlation_matrix_pearson)

        raise Exception("Should not be here :(")

    def closest(self, w, n=10):
        """
        Assumes the vectors have been normalized.
        """
        scores = self.m.dot(self.represent(w))
        return heapq.nlargest(n, list(zip(scores, self.iw)))
    

class SVDEmbedding(Embedding):
    """
    SVD embeddings.
    Enables controlling the weighted exponent of the eigenvalue matrix (eig).
    Context embeddings can be created with "transpose".
    """
    
    def __init__(self, path, normalize=True, eig=0.0, **kwargs):
        ut = np.load(path + '-u.npy', mmap_mode="c")
        s = np.load(path + '-s.npy', mmap_mode="c")
        vocabfile = path + '-vocab.pkl'
        self.iw = load_pickle(vocabfile)
        self.wi = {w:i for i, w in enumerate(self.iw)}
 
        if eig == 0.0:
            self.m = ut
        elif eig == 1.0:
            self.m = s * ut
        else:
            self.m = np.power(s, eig) * ut

        self.dim = self.m.shape[1]

        if normalize:
            self.normalize()

class GigaEmbedding(Embedding):
    def __init__(self, path, words=[], dim=300, normalize=True, **kwargs):
        seen = []
        vs = {}
        for line in open(path):
            split = line.split()
            w = split[0]
            if words == [] or w in words:
                if len(split) != dim+1:
                    continue
                seen.append(w)
                vs[w] = np.array(list(map(float, split[1:])), dtype='float32')
        self.iw = seen
        self.wi = {w:i for i,w in enumerate(self.iw)}
        self.m = np.vstack(vs[w] for w in self.iw)
        if normalize:
            self.normalize()


