from histwords_py3.ioutils import load_pickle
import numpy as np
from sklearn import preprocessing


def get_widx_yidx_arr(dict_data):
    n_words = len(list(dict_data.keys()))
    ex_obs_years = next(iter(dict_data.values()))
    n_decades = len(ex_obs_years)
    years = list(range(min(ex_obs_years.keys()),
                       max(ex_obs_years.keys())+1, 10))
    widx = {}
    yidx = {}
    for idx, word in enumerate(dict_data.keys()):
        widx[word] = idx
    for idx, year in enumerate(years):
        yidx[year] = idx
    arr = np.zeros((n_words, n_decades))
    return arr, widx, yidx, years

def load_prep_polysemy(netstats_filepath, metric, normalize=True):
    """ metric <str> can be 'wclust', 'sum', 'bclust', 'deg' """
    data = load_pickle(netstats_filepath)
    polysemy_proxy = data[metric]
    arr, widx, yidx, years = get_widx_yidx_arr(polysemy_proxy)
    for word in polysemy_proxy.keys():
        for year in years:
            arr[widx[word]][yidx[year]] = -polysemy_proxy[word][year]
            # the minus iverts vals because high clust coeff / deg means low polysemy
    min_vals = np.min(arr, axis=0)
    # prop for log transform
    polysemy_data = (arr - min_vals)+1e-8 # shift to have strictly positive vals
    polysemy_data = np.log10(polysemy_data)  # log transform
    if normalize:
        for year in years:
            yearly_median = np.median(polysemy_data[:, yidx[year]])
            polysemy_data[:, yidx[year]] -= yearly_median
        polysemy_data = preprocessing.scale(polysemy_data) # mean 0 std 1
    return polysemy_data, widx, yidx

def load_prep_deltas(cossim_filepath, log_transform=True, normalize=True):
    """
    deltas data -- semantic change between consequitive decades as cosine similarity scores
    """
    deltas_data = load_pickle(cossim_filepath)
    arr, widx, yidx, years = get_widx_yidx_arr(deltas_data)
    for word in deltas_data.keys():
        for year in years:
            try:
                val = np.log10(1 - deltas_data[word][year])
                # log_transform of cos dist (deltas data is cos sim)
            except KeyError:
                val = np.nan
            arr[widx[word]][yidx[year]] = val
    if normalize:
        arr = preprocessing.scale(arr) # mean 0 std 1
    return arr, widx, yidx

def load_prep_synonymity(synon_filepath, log_transform=True, normalize=True):
    synonymity = load_pickle(synon_filepath)
    arr, widx, yidx, years = get_widx_yidx_arr(synonymity)
    for word in synonymity.keys():
        for year in years:
            try:
                val = synonymity[word][year]
                if np.isclose(0.0, val):
                    val = 1
                if log_transform:
                    val = np.log10(val)
            except KeyError:
                val = np.nan
            arr[widx[word]][yidx[year]] = val
    if normalize:
        #arr = preprocessing.scale(arr) # mean 0 std 1
        for year in years:
            yearly_median = np.median(arr[:, yidx[year]])
            arr[:, yidx[year]] -= yearly_median
    return arr, widx, yidx

def load_prep_freqs(freqs_filepath, log_transform=True, normalize=True):
    freqs = load_pickle(freqs_filepath)
    arr, widx, yidx, years = get_widx_yidx_arr(freqs)
    for word in freqs.keys():
        for year in years:
            freq_val = freqs[word][year]
            if np.isclose(0.0, freq_val):  # missing data
                freq_val = np.nan  # avoid artificial log(0) setting val to inf
            arr[widx[word]][yidx[year]] = freq_val
    if log_transform:
        arr = np.log10(arr)
    if normalize:
        arr = preprocessing.scale(arr) # mean 0 std 1
    return arr, widx, yidx

