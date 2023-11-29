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
import sys
import argparse
import time
from threading import Thread
# import mdtex2html
import mindspore as ms
import gradio as gr
from fastchat.conversation import get_conv_template
from mindformers.tools import logger
from mindformers import AutoModel, AutoTokenizer, TextIteratorStreamer, AutoConfig, logger
from mindformers.models import BloomTokenizer, LlamaTokenizer

def get_model_and_tokenizer(model_config, tokenizer_name):
    """Get model and tokenizer instance"""
    # return AutoModel.from_config(model_config), AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)
    return AutoModel.from_config(model_config), tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_target', default="Ascend", type=str, choices=['Ascend', 'CPU'],
                        help='The target device to run, support "Ascend" and "CPU". Default: Ascend.')
    parser.add_argument('--device_id', default=0, type=int, help='Which device to run service. Default: 0.')
    parser.add_argument('--model', type=str, default="/home/ma-user/work/mindformers/configs/llama_ailab/predict_llama2_7b_pretrain.yaml", help='Which model to generate text.')
    parser.add_argument('--tokenizer', type=str, default='/home/ma-user/work/ckpts/llama2-7b-pretrain/')
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
    # 多轮对话
    # history = ""
    # conv = get_conv_template("vicuna_v1.1").copy()
    # conv.roles = ('User', 'MedChat')
    # roles = {"User": conv.roles[0], "MedChat": conv.roles[1]}
    # print("##### 欢迎使用MedChat #####")
    # while True:
    #     user_input = input("请输入(输入exit结束对话，输入clear清除历史对话)：\n")
    #     if user_input == "exit":
    #         sys.exit()
    #     elif user_input == "clear":
    #         conv.messages = []
    #         print("##### 开启新一轮对话 #####")
    #     else:
    #         conv.append_message(roles["User"], user_input)
    #         logger.info("history text: %s", conv.get_prompt())
    #         sample_input = tokenizer(conv.get_prompt())
    #         sample_output = model.generate(sample_input["input_ids"], max_length=512)
    #         sample_output_sen = tokenizer.decode(sample_output, skip_special_tokens=True)
    #         medchat_out = sample_output_sen[0].split(user_input)[-1]
    #         print("MedChat:" + medchat_out)
    #         conv.append_message(roles["MedChat"], medchat_out)
    # 单论对话
    inputs = ["你好！",
                "介绍一下大连理工大学",
                "写一首关于夏天的诗"]
    inputs_ids = tokenizer(inputs, max_length=config.seq_length, padding="max_length")["input_ids"]
    outputs = model.generate(inputs_ids, max_length=512)
    for output in outputs:
        print(tokenizer.decode(output))