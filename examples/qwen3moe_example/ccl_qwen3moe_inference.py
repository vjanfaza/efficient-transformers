# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils.constants import Constants

model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
"""
# For CB inference, set continuous_batching to True and add full_batch_size,mxfp6,mint8 argument in compile function
# We will use prompt_len=1 for compilation for both cb and non-cb inference
"""
ctx_len = 1024
batch_size = 1
comp_ctx_lengths = [128, 256, 512, 1024]

"""
# Prefill_ccl_len shows how many numbers in the comp_ctx_lengths list is related to prefilling and the rest would be for decoding. The default value is 1 means the first value is for prefilling and the rest are for decoding.
# In moe models with prefill_seq_len=1, we can pass prefill_ccl_len=0 to use all ccl values for both prefilling and decoding steps.
"""
prefill_ccl_len = 0

model = QEFFAutoModelForCausalLM.from_pretrained(
    model_name, comp_ctx_lengths=comp_ctx_lengths, prefill_ccl_len=prefill_ccl_len
)
model.compile(
    prefill_seq_len=1,
    ctx_len=ctx_len,
    batch_size=batch_size,
    num_cores=16,
    num_devices=4,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    mos=1,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
exec_info = model.generate(prompts=Constants.INPUT_STR, tokenizer=tokenizer)
