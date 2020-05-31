import os,sys
import numpy as np
import matplotlib.pyplot as plt
from histwords_py3.ioutils import load_pickle

projects_root = os.path.abspath(os.pardir)
project_root = os.path.join(projects_root, "jokar")
histwords_root = os.path.join(project_root, "histwords_py3")
coha_folder = os.path.join(project_root, "coha-lemma")
embeddings_folder = os.path.join(coha_folder, "sgns")
sys.path.insert(0, histwords_root)

from load_prep import load_prep_polysemy, load_prep_deltas
from load_prep import load_prep_synonymity, load_prep_freqs

polysemy_data, widx_poly, yidx_poly = load_prep_polysemy(
    coha_folder+'/netstats/full-nstop_nproper-top10000.pkl',
    'bclust', normalize=True)
deltas_data, widx_deltas, yidx_deltas = load_prep_deltas(
    coha_folder+'/volstats/vols.pkl', log_transform=True, normalize=True)
synonymity_data, widx_synon, yidx_synon = load_prep_synonymity(
    coha_folder+'/synonymity_10000.pkl', log_transform=True, normalize=True)
freqs_data, widx_freqs, yidx_freqs = load_prep_freqs(
    coha_folder+'/freqs.pkl', log_transform=True, normalize=False)

feature_dict = {'freqs' : lambda word, year: freqs_data[widx_freqs[word]][yidx_freqs[year]],
                'deltas' : lambda word, year:
                    deltas_data[widx_deltas[word]][yidx_deltas[year]],
                'polysemy' : lambda word, year:
                    polysemy_data[widx_poly[word]][yidx_poly[year]],
                'synonimity' : lambda word, year:
                    synonymity_data[widx_synon[word]][yidx_synon[year]]}

from histwords_py3.statutils.mixedmodels import make_data_frame
words = list(widx_poly.keys())
years = list(range(1850, 2000+1, 10))  # freqs available only from 1850

df = make_data_frame(words, years, feature_dict)
df.year=df.year.astype('category').cat.codes
import statsmodels.formula.api as smf
lmm = smf.mixedlm("deltas ~ year + polysemy + freqs", df, groups=df["word"])

# Additional experiments with synonymity
#lmm = smf.mixedlm("deltas ~ year + polysemy + freqs + synonimity", df, groups=df["word"])
#lmm = smf.mixedlm("deltas ~ year + polysemy + synonimity", df, groups=df["word"])
#lmm = smf.mixedlm("deltas ~ year + freqs + synonimity", df, groups=df["word"])

lmmf = lmm.fit()
print(lmmf.summary())
print(df.corr())

# PLOTS
# from plots import preview_semantic_displacement, plot_polysemy
# plot_years = list(range(1880, 2000+1, 10))
# preview_semantic_displacement(deltas_data, widx_deltas, yidx_deltas,
#                               plot_years, r'Log-transformed, normalized $\tilde{\Delta}^{( t )} ( w_i )$')
# plot_polysemy(polysemy_data, widx_poly, yidx_poly, plot_years, r'Normalized log-transformed polysemy scores')

# embeddings loading
# from histwords_py3.representations.sequentialembedding import SequentialEmbedding
# emb = SequentialEmbedding.load(embeddings_folder, years)

# loading dict data
#freqs_data_rawdict = load_pickle(coha_folder+'/freqs.pkl')
#synonymity_rawdict = load_pickle(coha_folder+'/synonymity_10000.pkl')
#polysemy_rawdict = load_pickle(coha_folder+'/netstats/full-nstop_nproper-top10000.pkl')

