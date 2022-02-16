"""Script to index a collection with ColBERT

Usage: indexing.py COLBERT_PATH QUERY_TSV COLLECTION_TSV

"""


import os
import sys
sys.path.insert(0, '../')
from docopt import docopt


from colbert.infra import Run, RunConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher


if __name__ == '__main__':
    args = docopt(__doc__)


    queries = args["QUERY_TSV"]
    collection = args["COLLECTION_TSV"]

    queries = Queries(path=queries)
    collection = Collection(path=collection)

    f'Loaded {len(queries)} queries and {len(collection):,} passages'

    with Run().context(RunConfig(nranks=4, experiment='codebert')):  # nranks specifies the number of GPUs to use.
        index_name = f'codebert.index'

    indexer = Indexer(checkpoint=args["COLBERT_PATH"])
    indexer.index(name=index_name, collection=collection, overwrite=True)

    searcher = Searcher(index=index_name)

    query = queries[30]   # or supply your own query

    print(f"#> {query}")

    # Find the top-3 passages for this query
    results = searcher.search(query, k=3)

    # Print out the top-k retrieved passages
    for passage_id, passage_rank, passage_score in zip(*results):
        print(f"\t [{passage_rank}] \t\t {passage_score:.1f} \t\t {searcher.collection[passage_id]}")