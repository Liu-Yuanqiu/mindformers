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
    res = tokenizer.encode("介绍一下大连理工大学")
    print(res)
    return AutoModel.from_config(model_config), tokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--device_target', default="Ascend", type=str, choices=['Ascend', 'CPU'],
                    help='The target device to run, support "Ascend" and "CPU". Default: Ascend.')
parser.add_argument('--device_id', default=0, type=int, help='Which device to run service. Default: 0.')
parser.add_argument('--model', type=str, default="/home/ma-user/work/mindformers/configs/llama_ailab/predict_llama2_7b_pretrain.yaml", help='Which model to generate text.')
parser.add_argument('--tokenizer', type=str, default='/home/ma-user/work/ckpts/llama2-7b-lora/')
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
streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=True)

# pre-build the network
sample_input = tokenizer("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n以下这句话通顺吗？“表1-1世界丙烯腈需求及预测（万吨）地区2001年2002年2003年2004年2005年2010年欧洲130130130125100100(C)1994-2022 China Academic Journal Electronic Publishing House.All rights reserved.http://www.cnki.net上海师范大学硕士学位论文第1章绪论中东303030303535亚洲230250270280315340拉丁美洲202025253025北美706045454070全世界约480约490约500约510约520约550国内生产情况自20世纪80年代以来，我国丙烯腈工业发展较快，从国外引进8套装置，全部采用美国BP公司技术，目前总生产能力约为410kt/a。”，请用是或者不是回答。\n\n### Response:")
sample_output = model.generate(sample_input["input_ids"], max_length=1024)
print(tokenizer.decode(sample_output))