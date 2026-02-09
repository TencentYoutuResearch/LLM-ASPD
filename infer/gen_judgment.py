import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import sys

import numpy as np
from tqdm import tqdm

from common import (
    load_questions,
    load_model_answers,
    load_judge_prompts,
    check_data,
    play_a_match_single,
    get_model_list,
    Judge,
    MatchPair,
    MatchSingle,
    NEED_REF_CATS,
)


def make_match(
    questions,
    models,
    model_answers,
    judge,
    baseline_model,
    ref_answers=None,
    multi_turn=False,
):
    """
    Create match pairs between each candidate model and a baseline model for all questions.

    Args:
        questions (list): List of question dicts, each with a "question_id" 
            and potentially "turns" for multi-turn cases.
        models (list): List of model names to be compared against the baseline.
        model_answers (dict): Mapping from model name to another dict of {question_id: answer}.
        judge (Judge): Judge object to evaluate model answers.
        baseline_model (str): Name of the baseline model to compare against.
        ref_answers (dict, optional): Optional mapping for reference answers.
        multi_turn (bool, optional): If True, only match multi-turn questions.

    Returns:
        list: List of MatchPair objects representing each comparison.
    """
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            q_id = q["question_id"]
            m_1 = models[i]
            m_2 = baseline_model
            if m_1 == m_2:
                continue
            a_1 = model_answers[m_1][q_id]
            a_2 = model_answers[baseline_model][q_id]
            if ref_answers is not None:
                ref = ref_answers[judge.model_name][q_id]
                match = MatchPair(
                    dict(q),
                    m_1,
                    m_2,
                    a_1,
                    a_2,
                    judge,
                    ref_answer=ref,
                    multi_turn=multi_turn,
                )
            else:
                match = MatchPair(
                    dict(q), m_1, m_2, a_1, a_2, judge, multi_turn=multi_turn
                )
            matches.append(match)
    return matches


def make_match_all_pairs(
    questions,
    models,
    model_answers,
    judge,
    baseline_model=None,
    ref_answers=None,
    multi_turn=False,
):
    """
    Create match pairs between all possible combinations of models for the given questions.

    Args:
        questions (list): List of question dicts.
        models (list): List of model names to compare in pairs.
        model_answers (dict): Mapping from model name to dict of {question_id: answer}.
        judge (Judge): Judge object to evaluate model answers.
        baseline_model (str, optional): Not used, for interface compatibility.
        ref_answers (dict, optional): Optional mapping for reference answers.
        multi_turn (bool, optional): If True, only match multi-turn questions.

    Returns:
        list: List of MatchPair objects representing each possible model pair comparison.
    """
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                q_id = q["question_id"]
                m_1 = models[i]
                m_2 = models[j]
                a_1 = model_answers[m_1][q_id]
                a_2 = model_answers[m_2][q_id]
                if ref_answers is not None:
                    ref = ref_answers[judge.model_name][q_id]
                    match = MatchPair(
                        dict(q),
                        m_1,
                        m_2,
                        a_1,
                        a_2,
                        judge,
                        ref_answer=ref,
                        multi_turn=multi_turn,
                    )
                else:
                    match = MatchPair(
                        dict(q), m_1, m_2, a_1, a_2, judge, multi_turn=multi_turn
                    )
                matches.append(match)
    return matches


def make_match_single(
    questions,
    models,
    model_answers,
    judge,
    baseline_model=None,
    ref_answers=None,
    multi_turn=False,
):
    """
    Create single matches for each model's answer to each question, for single-model evaluation.

    Args:
        questions (list): List of question dicts.
        models (list): List of model names.
        model_answers (dict): Mapping from model name to dict of {question_id: answer}.
        judge (Judge): Judge object to evaluate answers.
        baseline_model (str, optional): Not used, for interface compatibility.
        ref_answers (dict, optional): Optional mapping for reference answers.
        multi_turn (bool, optional): If True, only match multi-turn questions.

    Returns:
        list: List of MatchSingle objects for every model-question pair.
    """
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            q_id = q["question_id"]
            m = models[i]
            a = model_answers[m][q_id]
            if ref_answers is not None:
                ref = ref_answers[judge.model_name][q_id]
                matches.append(
                    MatchSingle(
                        dict(q), m, a, judge, ref_answer=ref, multi_turn=multi_turn
                    )
                )
            else:
                matches.append(MatchSingle(dict(q), m, a, judge, multi_turn=multi_turn))
    return matches


def make_judge_pairwise(judge_model, judge_prompts):
    """
    Construct a dictionary of Judge objects for various pairwise (two-answer) evaluation scenarios.

    Args:
        judge_model (str): The name or identifier of the judge model.
        judge_prompts (dict): Dictionary mapping prompt types to prompt templates.

    Returns:
        dict: Dictionary with Judge objects keyed by scenario types (e.g., "default", "math").
    """
    judges = {}
    judges["default"] = Judge(judge_model, judge_prompts["pair-v2"])
    judges["math"] = Judge(judge_model, judge_prompts["pair-math-v1"], ref_based=True)
    judges["default-mt"] = Judge(
        judge_model, judge_prompts["pair-v2-multi-turn"], multi_turn=True
    )
    judges["math-mt"] = Judge(
        judge_model,
        judge_prompts["pair-math-v1-multi-turn"],
        ref_based=True,
        multi_turn=True,
    )
    return judges


def make_judge_single(judge_model, judge_prompts):
    judges = {}
    judges["default"] = Judge(judge_model, judge_prompts["single-v1"])
    judges["math"] = Judge(judge_model, judge_prompts["single-math-v1"], ref_based=True)
    judges["default-mt"] = Judge(
        judge_model, judge_prompts["single-v1-multi-turn"], multi_turn=True
    )
    judges["math-mt"] = Judge(
        judge_model,
        judge_prompts["single-math-v1-multi-turn"],
        ref_based=True,
        multi_turn=True,
    )
    return judges


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--judge-file",
        type=str,
        default="benchmarks/judge_prompts.jsonl",
        help="The file of judge prompts.",
    )
    parser.add_argument("--judge-model", type=str, default="gpt-4")
    parser.add_argument("--baseline-model", type=str, default="gpt-3.5-turbo")
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
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument(
        "--first-n", type=int, help="A debug option. Only run the first `n` judgments."
    )
    args = parser.parse_args()

    question_file = f"benchmarks/{args.bench_name}/question.jsonl"
    answer_dir = f"APAR_BENCH_Qwen25-7B_RES/{args.bench_name}/model_answer"
    ref_answer_dir = f"benchmarks/{args.bench_name}/reference_answer"

    print(f'answer_dir:{answer_dir}')
    print(f'ref_answer_dir:{ref_answer_dir}')
    # Load questions
    questions = load_questions(question_file, None, None)

    # Load answers
    model_answers = load_model_answers(answer_dir)
    ref_answers = load_model_answers(ref_answer_dir)

    # Load judge
    judge_prompts = load_judge_prompts(args.judge_file)

    if args.first_n:
        questions = questions[: args.first_n]

    if args.model_list is None:
        models = get_model_list(answer_dir)
    else:
        models = args.model_list

    if args.mode == "single":
        judges = make_judge_single(args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_single
        output_file = (
            f"eval/{args.bench_name}/model_judgment/{args.judge_model}_single.jsonl"
        )
        make_match_func = make_match_single
        baseline_model = None
    else:
        judges = make_judge_pairwise(args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_pair
        output_file = (
            f"eval/{args.bench_name}/model_judgment/{args.judge_model}_pair.jsonl"
        )
        if args.mode == "pairwise-all":
            make_match_func = make_match_all_pairs
            baseline_model = None
        else:
            make_match_func = make_match
            baseline_model = args.baseline_model

    valid_models = check_data(questions, model_answers, ref_answers, models, judges)
    
    if not valid_models:
        print("Error: No valid models found after data check")
        sys.exit(1)
    
    print(f"Using {len(valid_models)} valid models: {valid_models}")

    question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
    question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

    # Make matches
    matches = []
    matches += make_match_func(
        question_default, valid_models, model_answers, judges["default"], baseline_model
    )
    matches += make_match_func(
        question_math,
        valid_models,
        model_answers,
        judges["math"],
        baseline_model,
        ref_answers,
    )
    matches += make_match_func(
        question_default,
        valid_models,
        model_answers,
        judges["default-mt"],
        baseline_model,
        multi_turn=True,
    )
    matches += make_match_func(
        question_math,
        valid_models,
        model_answers,
        judges["math-mt"],
        baseline_model,
        ref_answers,
        multi_turn=True,
    )
    # 去除已经跑过的matches
    map_data = set()
    input_file = (
        f"eval/{args.bench_name}/model_judgment/{args.judge_model}_single.jsonl"
    )
    import pandas as pd
    import os
    if os.path.isfile(input_file):
        print(f'reload from: {input_file}')
        df_all = pd.read_json(input_file, lines=True)
        for idx, d in df_all.iterrows():
            assert d['score'] != -1, f'{input_file}: [line {idx+1}]找到无效数据（score==-1）请删除： \n{d}'
            map_data.add(str(d["question_id"])+d["model"]+", ".join(d["judge"]))
        del df_all
        new_matches = []
        for m in matches:
            judge = m.judge
            set_id = str(m.question["question_id"])+m.model+", ".join([judge.model_name, judge.prompt_template["name"]])
            if set_id in map_data:
                continue
            print(f'add math: {set_id}')
            new_matches.append(m)
        matches = new_matches
    # exit(0)


    match_stat = {}
    match_stat["bench_name"] = args.bench_name
    match_stat["mode"] = args.mode
    match_stat["judge"] = args.judge_model
    match_stat["baseline"] = baseline_model
    match_stat["model_list"] = models
    match_stat["total_num_questions"] = len(questions)
    match_stat["total_num_matches"] = len(matches)
    match_stat["output_path"] = output_file

    # Show match stats and prompt enter to continue
    print("Stats:")
    print(json.dumps(match_stat, indent=4))
    # input("Press Enter to confirm...")

    # Play matches
    if args.parallel == 1:
        for match in tqdm(matches):
            play_a_match_func(match, output_file=output_file)
    else:

        def play_a_match_wrapper(match):
            return play_a_match_func(match, output_file=output_file)

        np.random.seed(0)
        np.random.shuffle(matches)
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        import multiprocessing
        if matches:
            with open(output_file, "a") as fout:
                with multiprocessing.Pool(processes=args.parallel) as executor:
                    for match in tqdm(
                        executor.imap_unordered(play_a_match_wrapper, matches), total=len(matches)
                    ):
                        if match:
                            fout.write(json.dumps(match) + "\n")

