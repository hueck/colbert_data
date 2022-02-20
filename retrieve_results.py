"""Build index, use the queries to search and map predictions to corresponding urls for later
use with the relevanceeval.py script from the CodeSearchNet Challenge.

More info about the CodeSearchNet Challenge can be found at:
https://github.com/github/CodeSearchNet#annotations
The script can be downloaded from:
https://github.com/github/CodeSearchNet/blob/76a006fda591591f196fd0aef1d6282af3135f71/src/relevanceeval.py

Usage:
    retrieve_results.py PATH_QUERIES PATH_COLLECTION PATH_FULL_DATA PATH_COLBERT OUT_DIR INDEX_NAME [--gpus=<n>]

Options:
    PATH_QUERIES        path to tsv file with queries as produced by convert_query_file.py
    PATH_COLLECTION     path to tsv file with collection as produced by build_triples.py
    PATH_FULL_DATA      path to tsv file with additional data fields of example contained in the collection
        (needed to map the predicted id to an url, as needed by the relevanceeval.py script)
    PATH_COLBERT        path to ColBERT checkpoint
    OUT_DIR             path to output directory
    INDEX_NAME          name of index (subdirectory with this name is created in OUT_DIR)
    --gpus=<n>          number of gpus to use [default: 1]
"""
from docopt import docopt
import pandas as pd
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher

if __name__ == '__main__':
    args = docopt(__doc__)

    queries = Queries(path=args["PATH_QUERIES"])
    collection = Collection(path=args["PATH_COLLECTION"])
    full_data = pd.read_csv(args["PATH_FULL_DATA"], delimiter="\t")

    f'Loaded {len(queries)} queries and {len(collection):,} passages'

    with Run().context(RunConfig(nranks=int(args["--gpus"]), experiment="colbert")):
        config = ColBERTConfig(index_path=args["OUT_DIR"] + "/index/" + args["INDEX_NAME"])
        # build index
        index_name = args["INDEX_NAME"]
        indexer = Indexer(checkpoint=args["PATH_COLBERT"], config=config)
        config = ColBERTConfig(index_path=args["OUT_DIR"] + "/" + args["INDEX_NAME"])
        indexer.index(name=index_name, collection=collection, overwrite=True)

        searcher = Searcher(index=index_name, config=config)
        result = searcher.search_all(queries, k=300).todict()

    language = "Java"
    # join query_id with urls from full data
    results = []
    for query_id, predictions in result.items():
        for collection_id, _, _ in predictions:
            # note that the query text is saved, not the id
            results.append((language, queries.data[query_id], full_data.loc[collection_id, "url"]))

    result_df = pd.DataFrame(results, columns=["language", "query", "url"])
    result_df.to_csv(args["OUT_DIR"] + "/predictions.csv", sep=",", index=False)
