"""Build triples from pkl-file of CodeSearchNet dataset.
Use function name + documentation as query, function body with masked function name as positive passage, and
random function as negative passage.
For now only works with Java functions.

Usage:
    build_triples.py IN_DIR OUT_DIR

"""
import os
import re
import string

import pandas as pd
from pandas import DataFrame, Series
from docopt import docopt
from tqdm import tqdm
import json_lines


def collect_data(in_dir: string) -> DataFrame:
    functions = []
    for dir_name in os.listdir(in_dir):
        if not os.path.isdir(f"{in_dir}/{dir_name}"):
            continue
        filenames = [filename for filename in os.listdir(f"{in_dir}/{dir_name}") if filename.endswith(".gz")]
        for file in filenames:
            print(f"extracting functions from {file}")
            with json_lines.open(f"{in_dir}/{dir_name}/{file}", 'rb') as line_generator:
                function_lines = [{"code": x["code"],
                                   "identifier": x["func_name"],
                                   "partition": x["partition"],
                                   "docstring": x["docstring"],
                                   } for x in line_generator]
                functions.extend(function_lines)
    df = pd.DataFrame.from_records(functions)
    return df


def process_functions(df: DataFrame) -> DataFrame:
    # remove package name in front of identifier (Observable.zipArray -> zipArray)
    df["clean_identifier"] = [clean_func_name(identifier) for identifier in
                              tqdm(df["identifier"], desc="Cleaning function identifiers")]
    df["code"] = [mask_function_name(str(code), identifier.split(".")[-1]) for code, identifier in
                  tqdm(zip(df["code"], df["identifier"]), total=len(df.index),
                       desc="Masking identifiers")]
    # remove any functions where the identifier was not masked properly
    df = df[df["code"].map(lambda x: " f(" in x)].copy()

    df["code"] = df["code"].apply(lambda x: x.replace("\t", " ").replace("\n", " ").replace("\r", " "))
    # df["code"] = df["code"].apply(lambda x: " ".join(x.splitlines()))
    # df["docstring"] = df["docstring"].apply(lambda x: " ".join(x.splitlines()))
    df["docstring"] = df["docstring"].apply(lambda x: x.replace("\t", " ").replace("\n", " ").replace("\r", " "))
    return df


def mask_function_name(function, func_name=None):
    """Apply regular expression to mask function name with f."""
    # ([\S\s]*\s)functionName\s*(\([^;]*\)\s*(?:throws\s*[\S\s]*)?{[\S\s]*})
    first = r"([\S\s]*\s)"
    second = r"\s*(\([^;]*\)\s*(?:throws\s*[\S\s]*)?{[\S\s]*})"
    re_expr = first + re.escape(func_name) + second
    result = re.sub(re_expr, r"\g<1> f\g<2>", function)
    return result


def clean_func_name(func_name):
    """Homogenize function names. (splitting at '.' and using last element, make lowercase, remove underscores)"""
    func_name = func_name.split(".")[-1]
    func_name = func_name.strip("_")
    return func_name


def add_negative_examples(data_set: DataFrame) -> DataFrame:
    """Generate negative examples."""
    # shuffle column of positive examples and use as negative examples
    data_set["negative_example"] = data_set["code"].sample(frac=1).values
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
    df = process_functions(df)

    # build datasets
    result = dict.fromkeys(df["partition"].unique())
    df["index"] = df.index
    for partition in df["partition"].unique():
        # split dataframe into train, test, validation set
        data_set = df[df["partition"] == partition].copy()
        # prepend docstring by identifier
        data_set["docstring"] = data_set["clean_identifier"] + " " + data_set["docstring"]
        # add negative examples to dataframes
        data_set = add_negative_examples(data_set)

        train_data = data_set.loc[:, ["docstring", "code", "negative_example"]]
        train_data.to_csv(args["OUT_DIR"] + f"/triples.{partition}.tsv", sep="\t", header=False, index=False)

        collection_data = data_set.loc[:, ["index", "code"]]
        collection_data.to_csv(args["OUT_DIR"] + f"/collection.{partition}.tsv", sep="\t", header=False, index=False)

        query_data = data_set.loc[:, ["index", "docstring"]]
        query_data.to_csv(args["OUT_DIR"] + f"/queries.{partition}.tsv", sep="\t", header=False, index=False)

        qrels = pd.DataFrame({"index": data_set["index"], "0": [0 for _ in range(0, len(data_set.index))],
                              "index2": data_set["index"], "1": [1 for _ in range(0, len(data_set.index))]})
        qrels.to_csv(args["OUT_DIR"] + f"/qrels.{partition}.tsv", sep="\t", header=False, index=False)
