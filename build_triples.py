"""Build triples from jsonlines files of CodeSearchNet dataset.
Use docstring as query, the corresponding function body as positive passage, and a random function as negative passage.
For now only works with Java functions.

More info about the data can be found at:
https://github.com/github/CodeSearchNet#data
More info about where to download the data at:
https://github.com/github/CodeSearchNet#downloading-data-from-s3

Usage:
    build_triples.py IN_DIR OUT_DIR

"""
import os
import string

import pandas as pd
from pandas import DataFrame, Series
from docopt import docopt
import json_lines


def collect_data(in_dir: string) -> DataFrame:
    """Read functions from all jsonl files in directory and return them as one dataframe."""
    functions = []
    filenames = [filename for filename in os.listdir(in_dir) if filename.endswith(".jsonl.gz")]
    if not filenames:
        raise FileNotFoundError("No compressed json-lines files found at: " + in_dir)
    for file in filenames:
        print(f"extracting functions from {file}")
        with json_lines.open(f"{in_dir}/{file}", 'rb') as line_generator:
            functions.extend([x for x in line_generator])
    df = pd.DataFrame.from_records(functions)
    return df


def clean_func_name(func_name):
    """Homogenize function names. (splitting at '.' and using last element, make lowercase, remove underscores)"""
    func_name = func_name.split(".")[-1]
    func_name = func_name.strip("_")
    return func_name


def add_negative_examples(data_set: DataFrame) -> DataFrame:
    """Generate negative examples."""
    # shuffle column of positive examples and use as negative examples
    data_set["negative_example"] = data_set["code"].sample(frac=1, random_state=42).values
    # make sure that every negative example is different from the positive example
    data_set["negative_example"] = data_set.apply(
        lambda row: get_negative_example(row, data_set)
        if row["code"] == row["negative_example"] else row["negative_example"], axis=1)
    return data_set


def get_negative_example(row: Series, data_set: DataFrame) -> string:
    """Return a negative example from the dataframe that is different from the positive example."""
    while row["negative_example"] == row["code"]:
        row["negative_example"] = data_set["code"].sample(1).iloc[0]
    return row["negative_example"]


if __name__ == '__main__':
    args = docopt(__doc__, argv=None, help=True)
    df = collect_data(args["IN_DIR"])

    df.drop(columns=["repo", "path", "sha", "partition"], inplace=True)

    # remove package reference in function name and strip underscores
    df["func_name"] = df["func_name"].apply(clean_func_name)

    # create index column
    df["index"] = df.index

    # remove tabs and newline symbols from code strings because ColBERT relies on these to parse tsv data
    df["code"] = df["code"].apply(lambda x: x.replace("\t", " ").replace("\n", " ").replace("\r", " "))
    df["docstring"] = df["docstring"].apply(lambda x: x.replace("\t", " ").replace("\n", " ").replace("\r", " "))

    # prepend docstring by identifier
    # df["docstring"] = df["func_name"] + " " + df["docstring"]

    # add negative examples to dataframes
    df = add_negative_examples(df)

    train_data = df.loc[:, ["docstring", "code", "negative_example"]]
    train_data.to_csv(args["OUT_DIR"] + "/triples.tsv", sep="\t", header=False, index=False)

    collection_data = df.loc[:, ["index", "code"]]
    collection_data.to_csv(args["OUT_DIR"] + "/collection.tsv", sep="\t", header=False, index=False)

    # save data for evaluation purposes and original code string for later use in retrieval application
    evaluation_data = df.loc[:, ["index", "original_string", "url", "language", "func_name"]]
    # note the escape char because orginal_string contains tab and newline symbols
    evaluation_data.to_csv(args["OUT_DIR"] + "/full_data.tsv", sep="\t", header=True, index=False, escapechar="\\")
