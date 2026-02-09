# Copyright 2024 OpenAccess AI Collective and the LlamaFactory team.
#
# This code is inspired by the OpenAccess AI Collective's axolotl library.
# https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/monkeypatch/utils.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DataCollatorForSeq2Seq

from ..extras.constants import AUDIO_PLACEHOLDER, IGNORE_INDEX, IMAGE_PLACEHOLDER
from ..extras.packages import is_pillow_available


if is_pillow_available():
    from PIL import Image


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from .template import Template


def prepare_4d_attention_mask(attention_mask_with_indices: "torch.Tensor", dtype: "torch.dtype") -> "torch.Tensor":
    r"""
    Expands the attention mask with indices from (batch_size, seq_len) to (batch_size, 1, seq_len, seq_len),
    while handles packed sequences and transforms the mask to lower triangular form to prevent future peeking.

    e.g.
    ```python
    # input
    [[1, 1, 2, 2, 2, 0]]
    # output
    [
        [
            [
                [o, x, x, x, x, x],
                [o, o, x, x, x, x],
                [x, x, o, x, x, x],
                [x, x, o, o, x, x],
                [x, x, o, o, o, x],
                [x, x, x, x, x, x],
            ]
        ]
    ]
    ```
    where `o` equals to `0.0`, `x` equals to `min_dtype`.
    """
    bsz, seq_len = attention_mask_with_indices.size()
    min_dtype = torch.finfo(dtype).min
    expanded_mask = attention_mask_with_indices[:, None, None, :].expand(bsz, 1, seq_len, seq_len)
    # Create a binary mask from the original mask where zeros remain zeros and all other values are set to one
    padding_mask = torch.where(expanded_mask != 0, 1, 0)
    # Create a block-diagonal mask.
    attention_mask_4d = torch.eq(expanded_mask, expanded_mask.transpose(-1, -2)).int() * padding_mask
    # Use the lower triangular mask to zero out the upper triangular part
    attention_mask_4d *= torch.tril(torch.ones((seq_len, seq_len), dtype=torch.long))
    # Invert the attention mask.
    attention_mask_4d = torch.where(attention_mask_4d != 0, torch.tensor(0, dtype=dtype), min_dtype)
    return attention_mask_4d

from typing import Any, Dict, List, Sequence, Tuple

@dataclass
class MultiModalDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    r"""
    Data collator that supports VLMs.

    Features should contain input_ids, attention_mask, labels, and optionally contain images, videos and audios.
    """

    template: Optional["Template"] = None
    processor: Optional["ProcessorMixin"] = None

    def __post_init__(self):
        if self.template is None:
            raise ValueError("Template is required for MultiModalDataCollator.")

    def __call__(self, feature_list: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        # 1) Gather raw multimodal batch data
        (
            batch_images,
            batch_videos,
            batch_audios,
            batch_imglens,
            batch_vidlens,
            batch_audlens,
            batch_input_ids,
        ) = self._gather_media_batches(feature_list)

        # 2) Optionally inject fake media and fake tokens (avoid zero3/fsdp hanging)
        fake_input_ids = []
        self._maybe_add_fake_images(
            fake_input_ids=fake_input_ids,
            batch_images=batch_images,
            batch_imglens=batch_imglens,
            batch_vidlens=batch_vidlens,
        )
        self._maybe_add_fake_audios(
            fake_input_ids=fake_input_ids,
            batch_audios=batch_audios,
            batch_audlens=batch_audlens,
        )

        # 3) If fake tokens were added, merge them into the first sample
        if fake_input_ids:
            self._apply_fake_tokens_to_first_feature(feature_list[0], fake_input_ids)
            batch_input_ids[0] = feature_list[0]["input_ids"]

        # 4) Build multimodal inputs for the model
        mm_inputs = self.template.mm_plugin.get_mm_inputs(
            batch_images,
            batch_videos,
            batch_audios,
            batch_imglens,
            batch_vidlens,
            batch_audlens,
            batch_input_ids,
            self.processor,
        )

        # 5) Attach token_type_ids back into features if provided by mm_inputs
        self._attach_token_type_ids(feature_list, mm_inputs)

        # 6) Call parent collator to tensorize and pad
        features: Dict[str, "torch.Tensor"] = super().__call__(feature_list)

        # 7) Optionally compute rope index (e.g., for qwen2vl mrope)
        self._maybe_add_rope_index(features, mm_inputs)

        # 8) Handle cross_attention_mask padding (e.g., for mllama when pad_to_multiple_of is enabled)
        self._maybe_pad_cross_attention_mask(features, mm_inputs)

        # 9) Merge multimodal tensors into final features
        features.update(mm_inputs)

        # 10) Special case: minicpmv inputs
        if "image_bound" in features:
            bsz, seq_length = features["input_ids"].shape
            features["position_ids"] = torch.arange(seq_length).long().repeat(bsz, 1)
            return {"data": features, "input_ids": features["input_ids"], "labels": features["labels"]}

        return features


    # ======================== Helper methods ========================

    def _gather_media_batches(
        self, feature_list: Sequence[Dict[str, Any]]
    ) -> Tuple[List[Any], List[Any], List[Any], List[int], List[int], List[int], List[List[int]]]:
        batch_images, batch_videos, batch_audios = [], [], []
        batch_imglens, batch_vidlens, batch_audlens, batch_input_ids = [], [], [], []

        for feat in feature_list:
            images = feat.pop("images", None) or []
            videos = feat.pop("videos", None) or []
            audios = feat.pop("audios", None) or []

            batch_images.extend(images)
            batch_videos.extend(videos)
            batch_audios.extend(audios)

            batch_imglens.append(len(images))
            batch_vidlens.append(len(videos))
            batch_audlens.append(len(audios))
            batch_input_ids.append(feat["input_ids"])

        return (
            batch_images,
            batch_videos,
            batch_audios,
            batch_imglens,
            batch_vidlens,
            batch_audlens,
            batch_input_ids,
        )


    def _maybe_add_fake_images(
        self,
        fake_input_ids: List[int],
        batch_images: List[Any],
        batch_imglens: List[int],
        batch_vidlens: List[int],
    ) -> None:
        # Avoid hanging when no images/videos while image_token is enabled
        if self.template.mm_plugin.image_token is None:
            return
        if sum(batch_imglens) != 0 or sum(batch_vidlens) != 0:
            return

        fake_messages = [{"role": "user", "content": IMAGE_PLACEHOLDER}]
        fake_images = [Image.new("RGB", (64, 64), (255, 255, 255))]

        fake_messages = self.template.mm_plugin.process_messages(
            fake_messages, fake_images, [], [], self.processor
        )
        _fake_input_ids = self.tokenizer.encode(fake_messages[0]["content"], add_special_tokens=False)
        _fake_input_ids, _ = self.template.mm_plugin.process_token_ids(
            _fake_input_ids, None, fake_images, [], [], self.tokenizer, self.processor
        )

        fake_input_ids.extend(_fake_input_ids)

        # Replace batch_images with fake image and set the first sample's length to 1
        batch_images.clear()
        batch_images.extend(fake_images)
        if batch_imglens:
            batch_imglens[0] = 1


    def _maybe_add_fake_audios(
        self,
        fake_input_ids: List[int],
        batch_audios: List[Any],
        batch_audlens: List[int],
    ) -> None:
        # Avoid hanging when no audios while audio_token is enabled
        if self.template.mm_plugin.audio_token is None:
            return
        if sum(batch_audlens) != 0:
            return

        fake_messages = [{"role": "user", "content": AUDIO_PLACEHOLDER}]
        fake_audios = [np.zeros(1600)]

        fake_messages = self.template.mm_plugin.process_messages(
            fake_messages, [], [], fake_audios, self.processor
        )
        _fake_input_ids = self.tokenizer.encode(fake_messages[0]["content"], add_special_tokens=False)
        _fake_input_ids, _ = self.template.mm_plugin.process_token_ids(
            _fake_input_ids, None, [], [], fake_audios, self.tokenizer, self.processor
        )

        fake_input_ids.extend(_fake_input_ids)

        # Replace batch_audios with fake audio and set the first sample's length to 1
        batch_audios.clear()
        batch_audios.extend(fake_audios)
        if batch_audlens:
            batch_audlens[0] = 1


    def _apply_fake_tokens_to_first_feature(self, first_feature: Dict[str, Any], fake_input_ids: List[int]) -> None:
        # Concatenate fake tokens to the first example according to padding side
        if self.tokenizer.padding_side == "right":
            first_feature["input_ids"] = first_feature["input_ids"] + fake_input_ids
            first_feature["attention_mask"] = first_feature["attention_mask"] + [0] * len(fake_input_ids)
            first_feature["labels"] = first_feature["labels"] + [IGNORE_INDEX] * len(fake_input_ids)
        else:
            first_feature["input_ids"] = fake_input_ids + first_feature["input_ids"]
            first_feature["attention_mask"] = [0] * len(fake_input_ids) + first_feature["attention_mask"]
            first_feature["labels"] = [IGNORE_INDEX] * len(fake_input_ids) + first_feature["labels"]


    def _attach_token_type_ids(self, feature_list: Sequence[Dict[str, Any]], mm_inputs: Dict[str, Any]) -> None:
        # Move token_type_ids from mm_inputs back to each sample feature (if present)
        if "token_type_ids" not in mm_inputs:
            return
        token_type_ids = mm_inputs.pop("token_type_ids")
        for i, feat in enumerate(feature_list):
            feat["token_type_ids"] = token_type_ids[i]


    def _maybe_add_rope_index(self, features: Dict[str, "torch.Tensor"], mm_inputs: Dict[str, Any]) -> None:
        # Compute rope index if model supports it
        if self.model is None or not hasattr(self.model, "get_rope_index"):
            return

        rope_index_kwargs = {
            "input_ids": features["input_ids"],
            "image_grid_thw": mm_inputs.get("image_grid_thw"),
            "video_grid_thw": mm_inputs.get("video_grid_thw"),
            "attention_mask": features["attention_mask"],
        }
        if "second_per_grid_ts" in mm_inputs:
            rope_index_kwargs["second_per_grid_ts"] = mm_inputs.get("second_per_grid_ts")

        features["position_ids"], features["rope_deltas"] = self.model.get_rope_index(**rope_index_kwargs)


    def _maybe_pad_cross_attention_mask(self, features: Dict[str, "torch.Tensor"], mm_inputs: Dict[str, Any]) -> None:
        # For mllama inputs when pad_to_multiple_of is enabled
        if "cross_attention_mask" not in mm_inputs:
            return
        cross_attention_mask = mm_inputs.pop("cross_attention_mask")
        seq_len = features["input_ids"].size(1)
        orig_len = cross_attention_mask.size(1)
        mm_inputs["cross_attention_mask"] = F.pad(cross_attention_mask, (0, 0, 0, 0, 0, seq_len - orig_len))

import os
import torch
def prepare_4d_para_attention_mask(attention_mask_with_indices: "torch.Tensor", dtype: "torch.dtype") -> "torch.Tensor":
    r"""Expand 2d attention mask to 4d attention mask.

    Expand the attention mask with indices from (batch_size, seq_len) to (batch_size, 1, seq_len, seq_len),
    handle packed sequences and transforms the mask to lower triangular form to prevent future peeking.

    Special handling for -1 indices: tokens with -1 index can be seen by all other tokens,
    but can only see tokens that come before them (maintaining causality).

    e.g.
    ```python
    # input
    [[1, 1, 2, 2, -1, 0]]
    # output
    [
        [
            [
                [o, x, x, x, x, x],  # 1只能看到自己
                [o, o, x, x, x, x],  # 1只能看到之前的1
                [x, x, o, x, x, x],  # 2只能看到自己
                [x, x, o, o, x, x],  # 2只能看到之前的2
                [o, o, o, o, o, x],  # -1可以看到之前的所有token
                [x, x, x, x, x, x],  # padding不看任何token
            ]
        ]
    ]
    ```
    where `o` equals to `0.0`, `x` equals to `min_dtype`.
    """
    bsz, seq_len = attention_mask_with_indices.size()
    min_dtype = torch.finfo(dtype).min
    zero_tensor = torch.tensor(0, dtype=dtype)

    # 获取每个分支块（连续的并行分支集合，因为可能出现多次并行块）的开始位置和结束位置
    branch_block = []
    
    for bi in range(attention_mask_with_indices.size(0)):
        sub_branch_block = []
        branch_st_pos = None
        branch_ed_pos = None
        for pi,val in enumerate(attention_mask_with_indices[bi]):
            if val>=1:
                if branch_st_pos is None:
                    branch_st_pos = pi
                else:
                    pass
            else:
                if branch_st_pos is None:
                    branch_st_pos = None
                    branch_ed_pos = None
                else:
                    branch_ed_pos = pi
                    sub_branch_block.append([branch_st_pos, branch_ed_pos])
                    branch_st_pos = None
                    branch_ed_pos = None
        if branch_st_pos is not None:
            sub_branch_block.append([branch_st_pos, len(attention_mask_with_indices[bi])])
        branch_block.append((bi, sub_branch_block))
    # for bb in branch_block:
    #     print('branch_blockbranch_blockbranch_blockbranch_block', bb)

    # padding 的位置
    padding_idxs = torch.nonzero(attention_mask_with_indices==0)
    # print(f'padding_idxs.shape: {padding_idxs.shape}')
    # print(f'padding_idxs: {padding_idxs}')
    # print(f'attention_mask_with_indices={attention_mask_with_indices}')

    # Create a non-padding mask.
    non_padding_mask = (attention_mask_with_indices != 0).unsqueeze(1).unsqueeze(2)
    
    # Create special mask for -1 indices (rows)
    special_indices_rows = (attention_mask_with_indices == -1).unsqueeze(1).unsqueeze(2)  # [bsz, 1, 1, seq_len]
    
    # Create special mask for -1 indices (columns)
    special_indices_cols = (attention_mask_with_indices == -1).unsqueeze(1).unsqueeze(3)  # [bsz, 1, seq_len, 1]
    
    # Create indices for comparison.
    indices = attention_mask_with_indices.unsqueeze(1).unsqueeze(2)  # [bsz, 1, 1, seq_len]
    indices_t = attention_mask_with_indices.unsqueeze(1).unsqueeze(3)  # [bsz, 1, seq_len, 1]
    
    # Create a lower triangular mask to maintain causality
    tril_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
    
    # Regular attention pattern: same indices can attend to each other
    regular_attn = (indices == indices_t) & non_padding_mask & tril_mask
    
    # Special attention pattern: 
    # 1. -1 token 可以看到所有之前的token (special_indices_rows & tril_mask)
    # 2. 所有token可以看到之前的-1 token (special_indices_cols & tril_mask)
    special_attn = ((special_indices_rows | special_indices_cols) & tril_mask & non_padding_mask)
    
    # Combine both attention patterns
    attention_mask_4d = regular_attn | special_attn
    
    # Invert the attention mask.
    attention_mask_4d = torch.where(attention_mask_4d, zero_tensor, min_dtype)
    attention_mask_4d[padding_idxs[:, 0], :, padding_idxs[:, 1], :] = min_dtype

    # 针对多个分支块进行mask修改，原本不同分支看不到其他分支包括之前的分支块（已经结束的）
    for bi, blocks in branch_block:
        for st, ed in blocks:
            attention_mask_4d[bi, :, st:ed, :st] = zero_tensor

    # print(f'attention_mask_4d.shape: {attention_mask_4d.shape}')
    # print_4d_bin_mask(attention_mask_4d)
    return attention_mask_4d


def mark_branches(input_list, para_start_id=151667, para_end_id=151668):
    """
    将输入列表中的数字根据151667和151668之间的位置进行分支标记
    
    Args:
        input_list: 输入的整数列表
    
    Returns:
        branch_marks: 标记后的列表，151667和151668之间的数字按分支编号，其余为-1
    """
    branch_marks = [-1] * len(input_list)
    # print(f'branch_marks={branch_marks}')
    branch_count = 0
    in_branch = False
    
    for i in range(len(input_list)):
        if input_list[i] == para_start_id:  # 分支开始
            branch_count += 1
            in_branch = True
        elif input_list[i] == para_end_id:  # 分支结束
            in_branch = False
        
        if in_branch:
            branch_marks[i] = branch_count
            # 标记分支结束符
            if i+1 < len(input_list) and input_list[i+1] == para_end_id:
                branch_marks[i+1] = branch_count
    
    # print(f'para_start_id={para_start_id}')
    # print(f'para_end_id={para_end_id}')
    # print(f'input_list={input_list}')
    # print(f'branch_marks={branch_marks}')
    return branch_marks




def get_async_group(input_list, para_start_id=151667, para_end_id=151668):
    """
    将输入列表中的数字根据151667和151668之间的位置进行分支标记
    
    Args:
        input_list: 输入的整数列表
    
    Returns:
        branch_marks: 标记后的列表，151667和151668之间的数字按分支编号，其余为-1
    """
    
    position_ids = []
    # find first para_step
    para_start_pos = []
    para_end_pos = []
    # fix_pos_ids = list(range(len(input_list)))
    # fix_pos_ids = 
    
    last_para_start_pos = None
    inner_para_pos = None
    in_branch = False
    max_pos_id = None
    inner_seq_pos = 0
    async_groups = []
    sub_groups = None
    for i,input_id in enumerate(input_list):
        if input_id == para_start_id:
            # 如果开始第一个分支
            if last_para_start_pos is None:
                sub_groups = []
                # 记录后续分支的开始位置编码
                last_para_start_pos = i
                # 重置内部编码计数器
                inner_para_pos = 0
                inner_para_pos += 1
            else:
                # 如果当前分支紧接着上一个分支结束
                if para_end_pos[-1]+1 == i:
                    # 重置内部编码计数器
                    inner_para_pos = 0
                    inner_para_pos += 1
                else:
                    # 当前所有分支已经结束，开启了新的分支
                    last_para_start_pos = i
                    # 分支内部编码
                    inner_para_pos = 0
                    inner_para_pos += 1
            para_start_pos.append(i)
        elif input_id == para_end_id:
            inner_para_pos += 1
            para_end_pos.append(i)
            sub_groups.append([para_start_pos[-1], para_end_pos[-1]+1])
        else:
            if len(para_start_pos)!=len(para_end_pos):
                inner_para_pos += 1
            else:
                last_para_start_pos = None # 分支结束了
                if sub_groups is not None:
                    async_groups.append(sub_groups)
                    sub_groups = None
    if sub_groups is not None:
        sub_groups.append([para_start_pos[-1], len(get_async_group)])
        async_groups.append(sub_groups)
        
    return async_groups

def fixrange_positions(input_list, para_start_id, para_end_id):
    agroups = get_async_group(input_list,para_start_id,para_end_id)
    fix_pos_ids = list(range(len(input_list)))
    cur_agroup_idx = 0
    cur_async_idx = agroups[cur_agroup_idx][0][0] if cur_agroup_idx<len(agroups) else None
    idx = 0
    pos_id = 0
    len_input_list = len(input_list)
    while idx < len_input_list:
        if idx == cur_async_idx:
            max_range_size = max(g[1]-g[0] for g in agroups[cur_agroup_idx])
            num_async = len(agroups[cur_agroup_idx])
            same_list = np.array(list(range(pos_id, pos_id+max_range_size)))
            for ai, (f,r) in enumerate(agroups[cur_agroup_idx]):
                fix_pos_ids[f:r] = (same_list[:r-f] + ai*max_range_size).tolist()
            
            idx = agroups[cur_agroup_idx][-1][1]
            pos_id += num_async*max_range_size
            cur_agroup_idx += 1
            cur_async_idx = agroups[cur_agroup_idx][0][0] if cur_agroup_idx<len(agroups) else None
        else:
            fix_pos_ids[idx] = pos_id
            pos_id += 1
            idx+=1
    return fix_pos_ids

import numpy as np

def mark_positions(input_list, para_start_id=151667, para_end_id=151668, suffix_mode="normal"):
    """
    Mark positions in input list based on paragraph boundaries
    
    Args:
        input_list: List of integers to process
        para_start_id: ID marking paragraph start
        para_end_id: ID marking paragraph end
        suffix_mode: Processing mode - "normal", "max_para", or "fixrange"
    
    Returns:
        List with position IDs marked according to paragraph structure
    """
    assert suffix_mode in ["normal", "max_para", "fixrange"]
    
    if suffix_mode == "fixrange":
        return fixrange_positions(input_list, para_start_id, para_end_id)
    
    # Initialize tracking variables
    state = {
        'position_ids': list(range(len(input_list))),
        'para_start_positions': [],
        'para_end_positions': [],
        'last_para_start': None,
        'inner_para_counter': None,
        'max_position': None,
        'outer_sequence_pos': 0
    }
    
    # Process each element
    for i, input_id in enumerate(input_list):
        if input_id == para_start_id:
            _handle_paragraph_start(i, state, suffix_mode)
        elif input_id == para_end_id:
            _handle_paragraph_end(i, state)
        else:
            _handle_regular_element(i, state, suffix_mode)
    
    return state['position_ids']


def _handle_paragraph_start(index, state, suffix_mode):
    """Handle paragraph start marker"""
    if state['last_para_start'] is None:
        # First paragraph in sequence
        _initialize_new_paragraph_sequence(index, state, suffix_mode)
    else:
        # Check if this is a continuation or new sequence
        if _is_continuation(index, state):
            _continue_paragraph_sequence(index, state)
        else:
            _start_new_paragraph_sequence(index, state)
    
    state['para_start_positions'].append(index)


def _handle_paragraph_end(index, state):
    """Handle paragraph end marker"""
    state['position_ids'][index] = state['last_para_start'] + state['inner_para_counter']
    state['max_position'] = max(state['position_ids'][index], 
                                state['max_position'] or 0)
    state['inner_para_counter'] += 1
    state['para_end_positions'].append(index)


def _handle_regular_element(index, state, suffix_mode):
    """Handle regular elements (not paragraph markers)"""
    if _is_inside_paragraph(state):
        # Inside a paragraph
        state['position_ids'][index] = state['last_para_start'] + state['inner_para_counter']
        state['max_position'] = max(state['position_ids'][index], 
                                   state['max_position'] or 0)
        state['inner_para_counter'] += 1
    else:
        # Outside paragraphs
        _handle_outside_paragraph(index, state, suffix_mode)


def _initialize_new_paragraph_sequence(index, state, suffix_mode):
    """Initialize the first paragraph in a sequence"""
    if suffix_mode == "max_para":
        state['last_para_start'] = state['outer_sequence_pos']
    else:  # normal mode
        state['last_para_start'] = index
    
    state['inner_para_counter'] = 0
    state['position_ids'][index] = state['last_para_start'] + state['inner_para_counter']
    state['max_position'] = state['position_ids'][index]
    state['inner_para_counter'] += 1


def _is_continuation(index, state):
    """Check if current paragraph continues from previous"""
    return (state['para_end_positions'] and 
            state['para_end_positions'][-1] + 1 == index)


def _continue_paragraph_sequence(index, state):
    """Continue existing paragraph sequence"""
    state['inner_para_counter'] = 0
    state['position_ids'][index] = state['last_para_start'] + state['inner_para_counter']
    state['inner_para_counter'] += 1


def _start_new_paragraph_sequence(index, state):
    """Start a completely new paragraph sequence"""
    state['last_para_start'] = index
    state['inner_para_counter'] = 0
    state['position_ids'][index] = state['last_para_start'] + state['inner_para_counter']
    state['max_position'] = state['position_ids'][index]
    state['inner_para_counter'] += 1


def _is_inside_paragraph(state):
    """Check if currently inside a paragraph"""
    return len(state['para_start_positions']) != len(state['para_end_positions'])


def _handle_outside_paragraph(index, state, suffix_mode):
    """Handle elements outside of paragraphs"""
    state['last_para_start'] = None  # Mark paragraph sequence as ended
    
    if suffix_mode == "max_para":
        if state['max_position'] is not None:
            state['outer_sequence_pos'] = state['max_position'] + 1
            state['max_position'] = None
        state['position_ids'][index] = state['outer_sequence_pos']
        state['outer_sequence_pos'] += 1
    elif suffix_mode == "normal":
        pass  # Keep original position
    else:
        raise RuntimeError(f'Bad suffix_mode={suffix_mode}')

def print_4d_bin_mask(mask):
    min_val = mask.min()
    max_val = mask.max()
    bin_mask = torch.ones_like(mask, dtype=torch.long)
    bin_mask[mask==min_val] = 0
    bsz = mask.shape[0]
    print_str = ''
    for i in range(bsz):
        print_str += f'BIN ATTENTION MASK[bz={i}]===========\n'
        for mi in range(bin_mask.shape[-2]):
            print_str += '['
            for mj in range(bin_mask.shape[-1]):
                print(f'{bin_mask[i,:,mi,mj].item()}',end='')
                print_str += f'{bin_mask[i,:,mi,mj].item()}'
            print_str += ']\n'
        print_str += '=============================\n'
    print(print_str)

has_para_print = False

# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained(r'../../data_ppl/para_model/vicuna-7b-v1.3-Apar')
tokenizer = None

from typing import Any, Dict, Sequence, Tuple
import os
import torch

@dataclass
class SFTDataCollatorWith4DAttentionMask(MultiModalDataCollatorForSeq2Seq):
    r"""
    Data collator for 4d attention mask.
    """

    block_diag_attn: bool = False
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"
    compute_dtype: "torch.dtype" = torch.float32

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        """
        Lower-complexity version of the collator forward that:
        - Normalizes base features via parent collator
        - Optionally prepares block-diagonal attention mask
        - Casts floating tensors to compute dtype
        - Optionally builds paragraph-aware attention and position ids based on env settings
        """
        global has_para_print, tokenizer  # keep parity with existing globals
        features = super().__call__(features)

        # 1) Optional block-diagonal attention mask preparation
        self._maybe_prepare_block_diag_mask(features)

        # 2) Cast all floating tensors to the desired compute dtype
        self._cast_float_tensors(features)

        # 3) Read and validate paragraph-related environment configuration
        para_cfg = self._read_para_env()
        if not para_cfg["enabled"]:
            # No paragraph-aware processing requested
            return features

        # 4) Log the paragraph mode once
        self._log_para_train_once(para_cfg)

        # 5) Build indices and positions for paragraph-aware masking/posids
        device = features["input_ids"].device
        indices, positions = self._build_para_indices_and_positions(
            input_ids=features["input_ids"],
            attn_mask=features["attention_mask"],
            st_token=para_cfg["st_token"],
            ed_token=para_cfg["ed_token"],
            suffix_mode=para_cfg["suffix_mode"],
        )

        # 6) Build the final 4D attention mask according to para_mask mode
        if para_cfg["mask_mode"] == "cantsee":
            # prepare_4d_para_attention_mask expects a tensor of indices and a dtype
            answer_mask_4d = prepare_4d_para_attention_mask(
                torch.tensor(indices, device=device),
                self.compute_dtype,
            ).to(device)
        else:
            raise RuntimeError(f"Unsupported para_mask={para_cfg['mask_mode']}")

        # 7) Attach computed tensors back to features
        features["position_ids"] = torch.tensor(positions, device=device)
        features["attention_mask"] = answer_mask_4d

        return features


    # --------------------------- Helpers (reusable, testable) ---------------------------

    def _maybe_prepare_block_diag_mask(self, features: Dict[str, Any]) -> None:
        """
        Convert 2D attention mask into a 4D block-diagonal mask if required.
        """
        if getattr(self, "block_diag_attn", False) and getattr(self, "attn_implementation", "") != "flash_attention_2":
            features["attention_mask"] = prepare_4d_attention_mask(
                features["attention_mask"],
                self.compute_dtype,
            )

    def _cast_float_tensors(self, features: Dict[str, Any]) -> None:
        """
        Cast all floating tensors in features to self.compute_dtype.
        """
        for key, value in list(features.items()):
            if torch.is_tensor(value) and torch.is_floating_point(value):
                features[key] = value.to(self.compute_dtype)

    def _read_para_env(self) -> Dict[str, Any]:
        """
        Read and normalize paragraph-related environment configuration.
        Returns a dict with:
        - enabled: bool
        - mask_mode: str | None
        - st_token: int
        - ed_token: int
        - suffix_mode: str
        """
        para_train = os.environ.get("PARA_TRAIN")
        enabled = bool(para_train and para_train.lower() == "true")
        if not enabled:
            return {"enabled": False}

        para_mask = os.environ.get("PARA_MASK")
        st = os.environ.get("PARA_ST_TOKEN")
        ed = os.environ.get("PARA_ED_TOKEN")
        suffix_mode = os.environ.get("POSID_MODE", "normal")

        if st is None or ed is None:
            raise ValueError("PARA_ST_TOKEN and PARA_ED_TOKEN must be set when PARA_TRAIN is true")

        return {
            "enabled": True,
            "mask_mode": para_mask,
            "st_token": int(st),
            "ed_token": int(ed),
            "suffix_mode": suffix_mode,
        }

    def _log_para_train_once(self, cfg: Dict[str, Any]) -> None:
        """
        Print the paragraph training configuration once.
        """
        global has_para_print
        if not has_para_print:
            print(
                f"~~~~~~~~Enable para_train with para_st_token={cfg['st_token']} "
                f"and para_ed_token={cfg['ed_token']}, "
                f"para_mask={cfg['mask_mode']}, suffix_mode={cfg['suffix_mode']}"
            )
            has_para_print = True

    def _build_para_indices_and_positions(
        self,
        input_ids: "torch.Tensor",
        attn_mask: "torch.Tensor",
        st_token: int,
        ed_token: int,
        suffix_mode: str,
    ) -> Tuple[list, list]:
        """
        Build:
        - indices: list of lists with paragraph indices masked by attention
        - positions: list of lists with paragraph-aware position ids

        Any failure on a sequence will be logged and a safe fallback (zeros) is used
        to keep shapes consistent.
        """
        indices = []
        positions = []

        for ids, mask in zip(input_ids, attn_mask):
            try:
                # Compute paragraph indices and apply the base attention mask
                ans_idx = torch.tensor(
                    mark_branches(ids, st_token, ed_token),
                    device=ids.device,
                    dtype=mask.dtype,
                )
                ans_idx = ans_idx * mask  # elementwise mask
                indices.append(ans_idx.tolist())

                # Compute position ids
                pos = mark_positions(ids, st_token, ed_token, suffix_mode=suffix_mode)
                positions.append(pos)
            except Exception as e:
                # Robust fallback to keep batch processing stable
                print(f"Error while building para indices/positions. input_ids={ids}. Error: {e}")
                raise e

        return indices, positions

    # def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
    #     global has_para_print,tokenizer
    #     features = super().__call__(features)
    #     if self.block_diag_attn and self.attn_implementation != "flash_attention_2":
    #         features["attention_mask"] = prepare_4d_attention_mask(features["attention_mask"], self.compute_dtype)

    #     for key, value in features.items():  # cast data dtype for paligemma
    #         if torch.is_tensor(value) and torch.is_floating_point(value):
    #             features[key] = value.to(self.compute_dtype)
    #     para_train = os.environ.get('PARA_TRAIN', None)
    #     para_mask = os.environ.get('PARA_MASK',None)
    #     para_st_token = os.environ.get('PARA_ST_TOKEN', None)
    #     para_ed_token = os.environ.get('PARA_ED_TOKEN', None)
    #     suffix_mode = os.environ.get('POSID_MODE', "normal")
    #     if para_train is not None and para_train.lower() == 'true':
    #         if not has_para_print:
    #             print(f'~~~~~~~~Enable para_train with para_st_token={para_st_token} '
    #                 f'and para_ed_token={para_ed_token}, '
    #                 f'para_mask={para_mask}, suffix_mode={suffix_mode}')
    #             has_para_print = True
    #         para_st_token = int(para_st_token)
    #         para_ed_token = int(para_ed_token)

    #         device = features['input_ids'].device
    #         indices = []
    #         position_ids = []
    #         for ids,mask in zip(features['input_ids'],features['attention_mask']):
    #             try:
    #                 answer_indice = torch.tensor(mark_branches(ids, para_st_token, para_ed_token)) 
    #                 answer_indice = answer_indice * mask
    #                 indices.append(answer_indice.tolist())
    #                 position = mark_positions(ids,para_st_token, para_ed_token, suffix_mode=suffix_mode)
    #             except Exception as e:
    #                 print(f'Error Inputids=>【{ids}】')
    #             position_ids.append(position)
    #         if para_mask is not None and para_mask.lower() == 'cantsee':
    #             #print(f'### PARA MASK:{para_mask}')
    #             answer_mask_4d = prepare_4d_para_attention_mask(torch.tensor(indices), self.compute_dtype).to(device)
    #         else:
    #             raise RuntimeError(f'不支持的para_mask={para_mask}')
    #         position_ids = torch.tensor(position_ids).to(device)
    #         features['position_ids'] = position_ids
    #         features['attention_mask'] = answer_mask_4d
    #     return features


@dataclass
class PairwiseDataCollatorWithPadding(MultiModalDataCollatorForSeq2Seq):
    r"""
    Data collator for pairwise data.
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []
        for key in ("chosen", "rejected"):
            for feature in features:
                target_feature = {
                    "input_ids": feature[f"{key}_input_ids"],
                    "attention_mask": feature[f"{key}_attention_mask"],
                    "labels": feature[f"{key}_labels"],
                    "images": feature["images"],
                    "videos": feature["videos"],
                    "audios": feature["audios"],
                }
                concatenated_features.append(target_feature)

        return super().__call__(concatenated_features)


@dataclass
class KTODataCollatorWithPadding(MultiModalDataCollatorForSeq2Seq):
    r"""
    Data collator for KTO data.
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        target_features = []
        kl_features = []
        kto_tags = []
        for feature in features:
            target_feature = {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
                "labels": feature["labels"],
                "images": feature["images"],
                "videos": feature["videos"],
                "audios": feature["audios"],
            }
            kl_feature = {
                "input_ids": feature["kl_input_ids"],
                "attention_mask": feature["kl_attention_mask"],
                "labels": feature["kl_labels"],
                "images": feature["images"],
                "videos": feature["videos"],
                "audios": feature["audios"],
            }
            target_features.append(target_feature)
            kl_features.append(kl_feature)
            kto_tags.append(feature["kto_tags"])

        batch = super().__call__(target_features)
        kl_batch = super().__call__(kl_features)
        batch["kl_input_ids"] = kl_batch["input_ids"]
        batch["kl_attention_mask"] = kl_batch["attention_mask"]
        batch["kl_labels"] = kl_batch["labels"]
        if "cross_attention_mask" in kl_batch:  # for mllama inputs.
            batch["kl_cross_attention_mask"] = kl_batch["cross_attention_mask"]
        if "token_type_ids" in kl_batch:
            batch["kl_token_type_ids"] = kl_batch["token_type_ids"]

        batch["kto_tags"] = torch.tensor(kto_tags)
        return batch
