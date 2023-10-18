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
import os
import sys
import json
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

def get_output(conv, model, tokenizer):
    conv.messages = []
    conv.append_message(conv.roles[0], sample)
    history = conv.get_prompt().replace("</s>", " ")
    sample_input = tokenizer(history)
    if len(sample_input["input_ids"])>=512:
        return "NONE"
    sample_output = model.generate(sample_input["input_ids"], max_length=args.seq_length)
    sample_output_sen = tokenizer.decode(sample_output, skip_special_tokens=True)
    print(sample_output_sen[0])
    response = sample_output_sen[0].split("### Response:")[-1]
    return response.strip()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_target', default="Ascend", type=str, choices=['Ascend', 'CPU'],
                        help='The target device to run, support "Ascend" and "CPU". Default: Ascend.')
    parser.add_argument('--use_parallel', default=False, type=bool)
    parser.add_argument('--device_id', default=0, type=int, help='Which device to run service. Default: 0.')
    parser.add_argument('--type', default="entity", type=str, help='support "entity" "rel" and "kg". Default: entity.')
    parser.add_argument('--model', type=str, default="/home/ma-user/work/mindformers/configs/llama_ailab/finetuen_llama2_7b_lora_kg_entity.yaml", help='Which model to generate text.')
    parser.add_argument('--tokenizer', type=str, default='/home/ma-user/work/ckpts/chinese-llama2-tokenizer')
    parser.add_argument('--checkpoint_path', type=str, default="/home/ma-user/work/ckpts/llama2-7b-lora/kg_entity/llama2_7b_lora_rank_0_1-333_2.ckpt", help='The path of model checkpoint.')
    parser.add_argument('--seq_length', default="512", type=int, help="Sequence length of the model. Default: 512.")
    parser.add_argument('--use_past', default=True, type=bool,
                        help='Whether to enable incremental inference. Default: False.')
    parser.add_argument('--input_file', default="/home/ma-user/work/data/kg/sens_1017.json", type=str)
    parser.add_argument('--output_file', default="/home/ma-user/work/data/kg/sens_entity_1017.json", type=str)
    args = parser.parse_args()

    ms.set_context(mode=ms.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)
    config = AutoConfig.from_pretrained(args.model)
    config.seq_length = args.seq_length
    config.use_past = args.use_past
    if args.checkpoint_path:
        config.checkpoint_name_or_path = args.checkpoint_path
    logger.info("Config: %s", config)
    model, tokenizer = get_model_and_tokenizer(config, args.tokenizer)

    prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    )
    example = {
        "entity": {"instruction": "提取出句子中的所有实体","input": "乙烯和丙烯主要来源于石油；甲醇来源广泛，煤、石油、天然气和生物质都可以作为甲醇的生产源头，目前甲醇生产的主要原料是煤炭和天然气"},
        "rel": {"instruction": "请提取句子中头实体和尾实体之间的关系","input": "头实体：乙烯 尾实体：石油 句子：乙烯和丙烯主要来源于石油；甲醇来源广泛，煤、石油、天然气和生物质都可以作为甲醇的生产源头，目前甲醇生产的主要原料是煤炭和天然气"},
        "kg": {"instruction": "提取出句子中所有的三元组","input": "乙烯和丙烯主要来源于石油；甲醇来源广泛，煤、石油、天然气和生物质都可以作为甲醇的生产源头，目前甲醇生产的主要原料是煤炭和天然气"}
    }

    conv = get_conv_template("vicuna_v1.1").copy()
    if args.type == "entity":
        sen = prompt.format_map(example["entity"])
    elif args.type == "rel":
        sen = prompt.format_map(example["rel"])
    else:
        sen = prompt.format_map(example["kg"])

    conv.append_message(conv.roles[0], sen)
    # print(conv)
    history = conv.get_prompt().replace("</s>", " ")
    # print(history)
    sample_input = tokenizer(history)
    sample_output = model.generate(sample_input["input_ids"], max_length=args.seq_length)
    sample_output_sen = tokenizer.decode(sample_output, skip_special_tokens=True)
    print(sample_output_sen[0])
    
    if os.path.isfile(args.input_file):
        inputs = json.load(open(args.input_file, 'r', encoding="utf-8"))
        outputs = []
        for i in inputs:
            id = i["id"]
            print("########### id:"+str(id)+"###########")
            if args.use_parallel and id%8!=args.device_id:
                continue
            if args.type == "entity":
                sen = i["sen"]
                example["entity"]["input"] = sen
                sample = prompt.format_map(example["entity"])
                response = get_output(conv, model, tokenizer)
                res = response.split(",")
                out = set()
                for r in res:
                    out.add(r.strip())
                out1 = list(out)
                out_res = ",".join(out1)
                print("### Response:"+out_res)
                outputs.append({
                    "id": id,
                    "input": sen,
                    "output": out_res
                })
            elif args.type=="rel":
                sen = i["input"]
                entity_str = i["output"]
                entitys = entity_str.split(",")
                entitys_num = len(entitys)
                rels = []
                kgs = []
                for i in range(entitys_num):
                    for j in range(entitys_num):
                        if i==j:
                            continue
                        else:
                            sen_input = "头实体："+entitys[i]+" 尾实体："+entitys[j]+" 句子："+sen
                            example["rel"]["input"] = sen_input
                            sample = prompt.format_map(example["rel"])
                            response = get_output(conv, model, tokenizer)
                            rels.append(response)
                            kgs.append("("+entitys[i]+","+response+","+entitys[j]+")")
                rels1 = ",".join(rels)
                kgs1 = ",".join(kgs)
                outputs.append({
                    "id": id,
                    "sen": sen,
                    "entity": entity_str,
                    "rel": rels1,
                    "kg": kgs1
                })
            else:
                example["kg"]["input"] = sen
                sample = prompt.format_map(example["kg"])
                response = get_output(conv, model, tokenizer)

        if args.use_parallel:
            args.output_file = os.path.join('/home/ma-user/work/mindformers/output/log', "rank_"+str(args.device_id), args.output_file.split("/")[-1])
        json.dump(outputs, open(args.output_file, "w", encoding="utf-8"), ensure_ascii=False)

        if args.use_parallel:
            data = []
            for i in range(8):
                d = json.load(open(os.path.join("/home/ma-user/work/mindformers/output/log", "rank_"+str(i), args.output_file.split("/")[-1]), "r", encoding="utf-8"))
                data += d
            print(len(data))
            json.dump(data, open(args.output_file, "w", encoding="utf-8"), ensure_ascii=False)