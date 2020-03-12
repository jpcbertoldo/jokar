import os
import requests
import zipfile

from tqdm import tqdm


dir_path = os.path.dirname(os.path.realpath(__file__))


class Corpora(object):

    ENG_ALL = "english-all"
    ENG_FICTION = "english-fiction"
    COHA = "coha"
    COHA_LEMMA = "coha-lemma"
    FRENCH = "french"
    GERMAN = "german"
    CHINESE = "chinese"

    LIST = [ENG_ALL, ENG_FICTION, COHA, COHA_LEMMA, FRENCH, GERMAN, CHINESE]

    URLS_SGNS = {
        ENG_ALL: """http://snap.stanford.edu/historical_embeddings/eng-all_sgns.zip""",
        ENG_FICTION: """http://snap.stanford.edu/historical_embeddings/eng-fiction-all_sgns.zip""",
        COHA: """http://snap.stanford.edu/historical_embeddings/coha-word_sgns.zip""",
        COHA_LEMMA: """http://snap.stanford.edu/historical_embeddings/coha-lemma_sgns.zip""",
        FRENCH: """http://snap.stanford.edu/historical_embeddings/fre-all_sgns.zip""",
        GERMAN: """http://snap.stanford.edu/historical_embeddings/ger-all_sgns.zip""",
        CHINESE: """http://snap.stanford.edu/historical_embeddings/chi-sim-all_sgns.zip""",
    }

    FOLDERS = {
        ENG_ALL: "eng-all_sgns",
        ENG_FICTION: "eng-fiction-all_sgns",
        COHA: "coha-word_sgns",
        COHA_LEMMA: "coha-lemma_sgns",
        FRENCH: "fre-all_sgns",
        GERMAN: "ger-all_sgns",
        CHINESE: "chi-sim-all_sgns",
    }

    @staticmethod
    def get_folder(corpus_name):
        return os.path.abspath(os.path.join(dir_path, Corpora.FOLDERS[corpus_name]))

    @staticmethod
    def get_zipfile_path(corpus_name):
        return os.path.abspath(os.path.join(dir_path, Corpora.FOLDERS[corpus_name] + ".zip"))


def download_embeddings(corpus_name):

    if corpus_name not in Corpora.LIST:
        raise ValueError(
            "`{}` is not a valid corpus name. Choose from {{{}}}".format(corpus_name, ", ".join(Corpora.LIST))
        )

    folder = Corpora.get_folder(corpus_name)
    if os.path.isdir(folder):
        print("Corpus already there!")
        return

    zip_filename = Corpora.get_zipfile_path(corpus_name)

    if os.path.isfile(zip_filename):
        print("Zip file already downloaded :)")
    else:
        print("Downloading the zip file...")
        try:
            url = Corpora.URLS_SGNS[corpus_name]
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(zip_filename, 'wb') as f:
                    size = int(r.headers['Content-length'])
                    chunk_size = 1024 * 1024 * 3  # #MB
                    nb_iter = size / chunk_size
                    for chunk in tqdm(r.iter_content(chunk_size=chunk_size), total=nb_iter):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
        except Exception as ex:
            if os.path.isfile(zip_filename):
                print("Deleting zip file because something went wrong!")
                os.remove(zip_filename)
            raise ex

    try:
        print("Unzipping to {}...".format(folder))
        with zipfile.ZipFile(zip_filename, 'r') as zf:
            zf.extractall(folder)
    except Exception as ex:
        raise ex
    else:
        print("Deleting zip file...")
        os.remove(zip_filename)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        print("Choose one of the following corpora: ")
        print(sorted(Corpora.LIST))
        exit(1)
    else:
        corpus = sys.argv[1]
        if corpus != 'all':
            download_embeddings(corpus)
        else:
            for corpus in Corpora.LIST:
                download_embeddings(corpus)
