import logging
import json
import os
import xlsxwriter
import torch
import tqdm

def read2jsonline(filename: str):
    """
    Read a JSON Lines (.jsonl) file.

    Args:
        filename (str): The path to the .jsonl file.

    Returns:
        list: A list of dictionaries, each representing a line in the file.
    """
    lines = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [json.loads(l) for l in lines if l.strip() != '']
    return lines

def write2jsonline(filename: str, lines):
    """
    Write data to a JSON Lines (.jsonl) file.

    Args:
        filename (str): The path to the output .jsonl file.
        lines (list): A list of dictionaries to write to the file.
    """
    print(f'[{filename}] write size: {len(lines)}')
    with open(filename, 'w', encoding='utf-8', newline='\n') as f:
        for i, l in enumerate(lines):
            json.dump(l, f, ensure_ascii=False, indent=None, separators=(',', ':'))
            f.write('\n')

def read2json(filename):
    """
    Read a JSON file.

    Args:
        filename (str): The path to the .json file.

    Returns:
        dict or list: The data loaded from the JSON file.
    """
    with open(filename, 'r', encoding='utf-8') as fr:
        return json.load(fr)

def write2json(filename,save_data):
    """
    Write data to a JSON file with indentation.

    Args:
        filename (str): The path to the output .json file.
        save_data (dict or list): The data to be written into the file.
    """
    print(f'filename: {filename}, write size: {len(save_data)}')
    with open(filename, 'w', encoding='utf-8') as fw:
        json.dump(save_data, fw, ensure_ascii=False, indent=2)

def number_to_column(n):
    """
    Convert a column number to its corresponding Excel column letter(s).

    Args:
        n (int): The column number (1-based).

    Returns:
        str: The corresponding Excel column label (e.g., A, B, ..., AA, AB, ...).
    """
    result = ""
    while n > 0:
        n -= 1
        result = chr(n % 26 + ord('A')) + result
        n //= 26
    return result

def coordinates_to_excel(x, y):
    """
    Convert (row, column) coordinates to Excel cell notation.

    Args:
        x (int): Excel row number (1-based).
        y (int): Excel column number (1-based).

    Returns:
        str: Excel-style cell (e.g., 'A1', 'B2', etc.).
    """
    column = number_to_column(y)
    return f"{column}{x}"

def write2excel(filename: str, json_data, field_names: list[str]):
    """
    Write structured data into an Excel (.xlsx) file.

    Args:
        filename (str): The output path for the Excel file.
        json_data (list[dict]): List of dictionaries containing the data.
        field_names (list[str]): List of field names (keys) representing columns to be included.
    """
    if not json_data:
        return
    print(f'filename: {filename}, write size: {len(json_data)}')
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()

    worksheet.write(coordinates_to_excel(1, 1), '序号')
    for i, name in enumerate(field_names):
        worksheet.write(coordinates_to_excel(1, i + 2), name)

    for i, d in enumerate(json_data, 2):
        for j, name in enumerate(field_names):
            worksheet.write(coordinates_to_excel(i, j + 2), d[name])
            worksheet.write(coordinates_to_excel(i, 1), i - 2)
    workbook.close()


import numpy as np
import time
import os
import random

def setup_seed(seed):
    """
    Set random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
