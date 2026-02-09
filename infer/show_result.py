"""
Usage:
python3 show_result.py --mode [single|pairwise-baseline|pairwise-all]
"""
import argparse
import pandas as pd
import numpy as np
import re
from collections import defaultdict

# 编译正则表达式
pattern = re.compile(r'(.+?)_(run\d+)')

def extract_prefix_and_run(s):
    """
    提取字符串中的前缀和实验编号
    :param s: 输入字符串，如 'weqe_weq-weqe_run0'
    :return: (prefix, run_id) 元组，如 ('weqe_weq-weqe', 'run0')
    """
    match = pattern.match(s)
    if match:
        prefix = match.group(1)
        run_id = match.group(2)
        return prefix, run_id
    else:
        return None, None

import json

from collections import defaultdict

import xlsxwriter
import sys
sys.path.append('..')
from misc import *

def display_result_single(args):
    """
    Display single-turn evaluation results for selected models from a JSONL file.

    Args:
        args: Argument object containing:
            - input_file (str or None): Path to the input file. Uses default if not provided.
            - bench_name (str): Name of the benchmark dataset.
            - judge_model (str): Name of the judge model.
            - model_list (list or None): List of models to filter. Uses all if not provided.

    This function:
        - Loads model evaluation data from the specified file.
        - Filters out invalid scores.
        - Calculates and prints model scores for each turn and average scores.
    """
    if args.input_file is None:
        input_file = (
            f"eval/{args.bench_name}/model_judgment/{args.judge_model}_single.jsonl"
        )
    else:
        input_file = args.input_file

    print(f"Input file: {input_file}")
    df_all = pd.read_json(input_file, lines=True)
    
    df = df_all[["model", "score", "turn"]]
    df = df[df["score"] != -1]

    if args.model_list is not None:
        df = df[df["model"].isin(args.model_list)]

    print("\n########## First turn ##########")
    df_1 = df[df["turn"] == 1].groupby(["model", "turn"]).mean()
    df_1 = df_1.sort_values(by="score", ascending=False)
    for index, row in df_1.iterrows():
        print(index[0], row['score'])

    if args.bench_name == "mt_bench":
        print("\n########## Second turn ##########")
        df_2 = df[df["turn"] == 2].groupby(["model", "turn"]).mean()
        for index, row in df_2.iterrows():
            print(index, row['score'])

        print("\n########## Average ##########")
        df_3 = df[["model", "score"]].groupby(["model"]).mean()
        df_3 = df_3.sort_values(by="score", ascending=False)
        for index, row in df_3.iterrows():
            print(index, row['score'])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--judge-model", type=str, default="gpt-4")
    parser.add_argument("--baseline-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["pairwise-baseline", "pairwise-all", "single"],
        help=(
            "Evaluation mode. "
            "`pairwise-baseline` runs pairwise comparision against a baseline. "
            "`pairwise-all` runs pairwise comparision between all pairs. "
            "`single` runs single answer grading."
        ),
    )
    args = parser.parse_args()

    if args.mode == "single":
        display_result_func = display_result_single
    else:
        if args.mode == "pairwise-all":
            args.baseline_model = None
        display_result_func = display_result_pairwise

    print(f"Mode: {args.mode}")
    display_result_func(args)

