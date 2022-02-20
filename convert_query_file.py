"""Format the queries from the annotationstore.csv from the Code Search Net Challenge so that they can be used
with ColBERT.

The file can be found at:
https://github.com/github/CodeSearchNet#evaluation
https://github.com/github/CodeSearchNet/blob/76a006fda591591f196fd0aef1d6282af3135f71/resources/annotationStore.csv

Usage:
    convert_query_file.py PATH_ANNOTATIONSTORE_FILE OUT_DIR

"""

import pandas as pd
from docopt import docopt

if __name__ == '__main__':
    args = docopt(__doc__, argv=None, help=True)
    store = pd.read_csv(args["PATH_ANNOTATIONSTORE_FILE"], delimiter=",")

    # for now only select Java queries
    language = "Java"
    store = set(store.loc[store.Language == language]["Query"])
    store = pd.DataFrame(data=store, columns=["Query"])
    store["index"] = store.index

    # ColBERT requires the index to be equal with the line number in the file starting from 0
    store.reset_index(inplace=True)

    # store queries to tsv (an index column is added automatically)
    store.to_csv(args["OUT_DIR"] + "/queries.tsv", columns=["Query"], sep="\t", header=False)
