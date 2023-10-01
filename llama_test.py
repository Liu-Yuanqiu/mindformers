# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Run chat web demo.
"""
import argparse
import time
from threading import Thread
# import mdtex2html
import mindspore as ms
import gradio as gr

from mindformers import AutoModel, AutoTokenizer, TextIteratorStreamer, AutoConfig, logger
from mindformers.models import BloomTokenizer, LlamaTokenizer

def get_model_and_tokenizer(model_config, tokenizer_name):
    """Get model and tokenizer instance"""
    # return AutoModel.from_config(model_config), AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)
    return AutoModel.from_config(model_config), tokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--device_target', default="Ascend", type=str, choices=['Ascend', 'CPU'],
                    help='The target device to run, support "Ascend" and "CPU". Default: Ascend.')
parser.add_argument('--device_id', default=0, type=int, help='Which device to run service. Default: 0.')
parser.add_argument('--model', type=str, default="/home/ma-user/work/mindformers/configs/llama_ailab/predict_llama2_7b_pretrain.yaml", help='Which model to generate text.')
parser.add_argument('--tokenizer', type=str, default='/home/ma-user/work/ckpts/chinese-llama2-tokenizer/')
parser.add_argument('--checkpoint_path', type=str, default="/home/ma-user/work/ckpts/llama2-7b-pretrain/llama2-7b-pretrain.ckpt", help='The path of model checkpoint.')
parser.add_argument('--seq_length', default="512", type=int, help="Sequence length of the model. Default: 512.")
parser.add_argument('--use_past', default=False, type=bool,
                    help='Whether to enable incremental inference. Default: False.')
args = parser.parse_args()

ms.set_context(mode=ms.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)
config = AutoConfig.from_pretrained(args.model)
config.seq_length = args.seq_length
config.use_past = args.use_past
if args.checkpoint_path:
    config.checkpoint_name_or_path = args.checkpoint_path
logger.info("Config: %s", config)
model, tokenizer = get_model_and_tokenizer(config, args.tokenizer)

# pre-build the network
sample_input = tokenizer("介绍一下大连理工大学")
print(sample_input)
sample_output = model.generate(sample_input["input_ids"], max_length=512)
print(tokenizer.decode(sample_output))