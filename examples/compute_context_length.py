# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

## In this example, you can run a model for static and continuous batching with different Compute-Context-Length (CCL) inputs. ##

from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

## Using optional variable comp_ctx_lengths variable you can pass a list of context lengths. It will run the model with default context length if comp_ctx_lengths=None. ##
##       - The first Prefill_ccl_len numbers in this list are the context lengths that will be used during prefilling. ##
##       - During the decoding process, based on the position_id or cache index it will work with the specific compute-context-length in the list. It will start from a proper compute-context-length in the list based on input prompt length and will gradually increase the compute-context-length if the cache index passes the current compute-context-length. ##


ctx_len = 2048
comp_ctx_lengths_prefill = [256]
comp_ctx_lengths_decode = [512, 1024, ctx_len]

model_name = "Qwen/Qwen2.5-7B"
model = QEFFAutoModelForCausalLM.from_pretrained(
    model_name,
    continuous_batching=True,
    comp_ctx_lengths_prefill=comp_ctx_lengths_prefill,
    comp_ctx_lengths_decode=comp_ctx_lengths_decode,
    ctx_len=ctx_len,
)

# model compilation for either continuous or static batching. For continuous batching full_batch_size is needed.
model.compile(
    prefill_seq_len=128,
    ctx_len=ctx_len,
    num_cores=16,
    num_devices=4,
    full_batch_size=1,
    mxint8_kv_cache=True,
    mxfp6_matmul=True,
)

# Create tokenizer and run model.generate and passes the input prompts to it.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.generate(
    prompts=[
        "My name is ",
    ],
    tokenizer=tokenizer,
)
