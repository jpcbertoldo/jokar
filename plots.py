
import matplotlib.pyplot as plt

def preview_semantic_displacement(deltas_data, widx, yidx, years, plot_title):
    """
    deltas_data is np.array with delta data on decades (cols) for each word (row)
    widx <dict> points to the row index for a word (key in a dict)
    yidx <dict> points to the col index for a year starting a decade
    """
    lines = ["--","-","-."]
    # semantically most changed words
    for i, word in enumerate(['gay', 'actually']):
        d = []
        for year in years:
            d.append((deltas_data[widx[word]][yidx[year]]))
        plt.plot(years, d, ls=f'{lines[i]}', c='r', label=f'{word}', alpha=.8)
    # regular words
    for i, word in enumerate(['dog', 'cat']):
        d = []
        for year in years:
            d.append((deltas_data[widx[word]][yidx[year]]))
        plt.plot(years, d, ls=f'{lines[i]}', c='b', label=f'{word}', alpha=.8)
    plt.title(plot_title)
    plt.legend()
    plt.savefig(f'{plot_title}')
    plt.clf()

def plot_polysemy(poly_data, widx, yidx, years, plot_title):
    lines = ["--","-","-."]
    most_polysemous = ['yet', 'bank']
    least_polysemous = ['thirties', 'mom']
    for i, word in enumerate(most_polysemous):
        poly = []
        for year in years:
            poly.append(poly_data[widx[word]][yidx[year]])
        plt.plot(years, poly, ls=f'{lines[i]}', c='r', label=f'{word}', alpha=.8)
    for i, word in enumerate(least_polysemous):
        poly = []
        for year in years:
            poly.append(poly_data[widx[word]][yidx[year]])
        plt.plot(years, poly, ls=f'{lines[i]}', c='b', label=f'{word}', alpha=.8)
    plt.title(plot_title)
    plt.legend()
    plt.savefig(f'{plot_title}')
    plt.clf()

