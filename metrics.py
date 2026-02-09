

def compute_parallel_efficiency(total_branch_content_tokens, 
    total_title_tokens, total_text_tokens):
    """
    Compute the parallel efficiency.
    
    Parallel efficiency is calculated as the proportion of non-title tokens 
    in all branch contents. Returns 0 if total_title_tokens is zero.
    
    Args:
        total_branch_content_tokens (int): The total number of tokens in all branch contents.
        total_title_tokens (int): The total number of tokens in all titles.
        total_text_tokens (int): The total number of tokens in all texts (unused in this function).
    
    Returns:
        float: The parallel efficiency value.
    """
    return ((total_branch_content_tokens-total_title_tokens) / 
                (total_branch_content_tokens) if total_title_tokens else 0)

def compute_parallel_degree(total_branch_content_tokens, 
    total_title_tokens, total_text_tokens):
    """
    Compute the parallel degree.
    
    Parallel degree is defined as the ratio of total branch content tokens 
    to the total text tokens. Returns 0 if total_text_tokens is zero.
    
    Args:
        total_branch_content_tokens (int): The total number of tokens in all branch contents.
        total_title_tokens (int): The total number of tokens in all titles (unused in this function).
        total_text_tokens (int): The total number of tokens in all texts.
    
    Returns:
        float: The parallel degree value.
    """
    return total_branch_content_tokens / total_text_tokens if total_text_tokens else 0

def get_total_serial_tokens(parsed):
    """
    Calculate the total number of tokens in synchronous (serial) parts.
    
    For each item in parsed input:
      - If the item is of type "sync", sum the length of its content.
      - If the item is not "sync", sum the lengths of contents in all inner branches.
    
    Args:
        parsed (list): A list of items where each item is a dict with keys "type" and "content".
    
    Returns:
        int: Total number of tokens in all synchronous parts.
    """
    return sum(
                len(item["content"]) if item["type"]=="sync"
                else sum(len(branch["content"]) for branch in item["content"])
                for item in parsed
            )

def get_asyncgroups(parsed):
    """
    Extract all asynchronous groups from the parsed input.
    
    Args:
        parsed (list): A list of items where each item is a dict with keys "type" and "content".
    
    Returns:
        list: A list of items with type "async".
    """
    return [item for item in parsed if item["type"] == "async"]

def count_parallel_info(parsed, asyncgroups):
    branch_num_list = [len(item["content"]) for item in asyncgroups]
    avg_branch = sum(branch_num_list) / len(branch_num_list)
    branch_count = sum(branch_num_list)

    total_title_tokens = sum(
        sum(len(branch["title"]) for branch in item["content"])
        for item in asyncgroups
    )
    total_branch_content_tokens = sum(
        sum(len(branch["content"]) for branch in item["content"])
        for item in asyncgroups
    )
    # Note that, each branch content is 'title: content' for inference, 
    # which containis title tokens
    total_text_tokens = sum(
        len(item["content"]) if item["type"]=="sync"
        else sum(len(branch["content"]) for branch in item["content"])
        for item in parsed
    )
    return (avg_branch, branch_count, total_branch_content_tokens,
         total_title_tokens, total_text_tokens)

def make_metrics_pack(can_parallel_count, cannot_parallel_count,
    avg_branch_counts,parallel_degrees,parallel_efficiencies,
    total_branch_content_tokens,  total_title_tokens, total_text_tokens):
    """
    Aggregate computed metrics into a dictionary for reporting or downstream analysis.

    If there is no parallel data, 'abn', 'apd', and 'ape' will be set to 0.
    Otherwise, computes averages for branch number, degree, and efficiency.

    Args:
        can_parallel_count (int): Number of samples with parallelizable async groups.
        cannot_parallel_count (int): Number of samples without async groups.
        avg_branch_counts (list of float): List of average branch numbers per group.
        parallel_degrees (list of float): List of parallel degree values.
        parallel_efficiencies (list of float): List of parallel efficiency values.
        total_branch_content_tokens (int): Total tokens in all branch contents.
        total_title_tokens (int): Total tokens in all branch titles.
        total_text_tokens (int): Total tokens in all text.

    Returns:
        dict: Dictionary containing all aggregated metrics.
    """
    ret = {
        "can_parallel_count": can_parallel_count,
        "cannot_parallel_count": cannot_parallel_count,
        "total_branch_content_tokens": total_branch_content_tokens,
        "total_title_tokens": total_title_tokens,
        "total_text_tokens": total_text_tokens,
    }
    if not avg_branch_counts:
        ret.update({
            "abn": 0,
            "apd": 0,
            "ape": 0,
        })
    else:
        ret.update({
            "abn": sum(avg_branch_counts) / len(avg_branch_counts),
            "apd": sum(parallel_degrees) / len(parallel_degrees),
            "ape": sum(parallel_efficiencies) / len(parallel_efficiencies),
        })
    return ret

def analyze_asyncgroups(all_parsed_results, tokenizer):
    """
    Analyze a collection of parsed results for inference scenarios 
        where branch content is 'title: content'.

    Iterates through the parsed results, identifies async groups, 
        and computes metrics on branch structure and parallelization statistics.

    Args:
        all_parsed_results (list): List of parsed samples (one sample is a list of items).
        tokenizer: Tokenizer instance (not used in this variant).

    Returns:
        dict: Dictionary of aggregated metrics (from make_metrics_pack).
    """
    can_parallel_count = 0
    cannot_parallel_count = 0
    total_branch_content_tokens, total_title_tokens = 0,0
    avg_branch_counts = []
    parallel_degrees = []
    parallel_efficiencies = []
    branch_counts = []
    for parsed in all_parsed_results:
        asyncgroups = get_asyncgroups(parsed)
        if asyncgroups:
            can_parallel_count += 1
            (avg_branch, branch_count, total_branch_content_tokens,
                total_title_tokens, total_text_tokens) =  count_parallel_info(parsed, asyncgroups)
            avg_branch_counts.append(avg_branch)
            branch_counts.append(branch_count)

            parallel_efficiencies.append(compute_parallel_efficiency(total_branch_content_tokens, 
                total_title_tokens, total_text_tokens))
            parallel_degrees.append(compute_parallel_degree(total_branch_content_tokens, 
                total_title_tokens, total_text_tokens))
        else:
            cannot_parallel_count += 1
            total_text_tokens = sum(
                len(item["content"]) if item["type"]=="sync"
                else sum(len(branch["content"]) for branch in item["content"])
                for item in parsed
            )
    return make_metrics_pack(can_parallel_count, cannot_parallel_count,
        avg_branch_counts,parallel_degrees,parallel_efficiencies,
        total_branch_content_tokens,  total_title_tokens, total_text_tokens)


def get_total_serial_tokens_text(parsed, tokenizer):
    return sum(
                len(tokenizer.encode(item["content"])) if item["type"]=="sync"
                else sum(len(tokenizer.encode(branch["content"])) for branch in item["content"])
                for item in parsed
            )

def count_parallel_info_text(parsed, asyncgroups, tokenizer):
    branch_num_list = [len(item["content"]) for item in asyncgroups]
    avg_branch = sum(branch_num_list) / len(branch_num_list)
    branch_count = sum(branch_num_list)

    total_branch_content_tokens = sum(
        sum(len(tokenizer.encode(branch["content"])) for branch in item["content"])
        for item in asyncgroups
    )
    total_title_tokens = sum(
        sum(len(tokenizer.encode(branch["title"])) for branch in item["content"])
        for item in asyncgroups
    )
    total_text_tokens = sum(
            len(tokenizer.encode(item["content"])) for item in parsed if item["type"]=="sync" 
        ) + total_branch_content_tokens + total_title_tokens
    return (avg_branch, branch_count, total_branch_content_tokens,
         total_title_tokens, total_text_tokens)


def analyze_asyncgroups_text(all_parsed_results, tokenizer):
    '''
        Use for Data PPL, where each branch content is 'content'
    '''
    can_parallel_count = 0
    cannot_parallel_count = 0
    avg_branch_counts = []
    parallel_degrees = []
    parallel_efficiencies = []
    branch_counts = []
    total_branch_content_tokens, total_title_tokens = 0,0
    for parsed in all_parsed_results:
        asyncgroups = get_asyncgroups(parsed)
        if asyncgroups:
            can_parallel_count += 1
            # branch_num_list = [len(item["content"]) for item in branchgroups]
            # avg_branch = sum(branch_num_list) / len(branch_num_list)
            (avg_branch, branch_count, total_branch_content_tokens, total_title_tokens, 
                total_text_tokens) =  count_parallel_info_text(parsed, asyncgroups, tokenizer)
            avg_branch_counts.append(avg_branch)
            branch_counts.append(branch_count)

            parallel_efficiencies.append(compute_parallel_efficiency(total_branch_content_tokens, 
                total_title_tokens, total_text_tokens))
            parallel_degrees.append(compute_parallel_degree(total_branch_content_tokens, 
                total_title_tokens, total_text_tokens))
        else:
            cannot_parallel_count += 1
            total_text_tokens = sum(
                len(tokenizer.encode(item["content"])) for item in parsed if item["type"]=="sync" 
            ) + total_branch_content_tokens + total_title_tokens
    return make_metrics_pack(can_parallel_count, cannot_parallel_count,
        avg_branch_counts,parallel_degrees,parallel_efficiencies,
        total_branch_content_tokens, total_title_tokens, total_text_tokens)

if __name__ == '__main__':
    # test
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_path = 'train/saves/Vicuna13-7B/full/Vicuna_ASPD'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    mock_list = [
        { 'type':'async', 'content': [
            {'content': 'ABC 123', 'title': '1'},
            {'content': 'EFG 789', 'title': '2'},
            {'content': 'HIJ', 'title': '3'},
            ]
        },
        { 'type': 'sync', 'content': '123'},
        { 'type': 'sync', 'content': '456'},
        { 'type':'async', 'content': [
            {'content': 'QWE', 'title': '4'},
            {'content': 'RTY', 'title': '5'},
            {'content': 'YUI', 'title': '6'},
            {'content': 'YUI', 'title': '6'},
            ]
        },
    ]
    print(analyze_asyncgroups_text([mock_list], tokenizer))
    mock_list = [
        { 'type':'async', 'content': [
            {'content': 'ABC', 'title': '1'},
            {'content': 'EFG', 'title': '2'},
            {'content': 'HIJ', 'title': '3'},
            ]
        },
        { 'type': 'sync', 'content': '123'},
        { 'type': 'sync', 'content': '456'},
        { 'type':'async', 'content': [
            {'content': 'QWE', 'title': '4'},
            {'content': 'RTY', 'title': '5'},
            {'content': 'YUI', 'title': '6'},
            {'content': 'YUI', 'title': '6'},
            ]
        },
    ]
    print(analyze_asyncgroups([mock_list], tokenizer))

