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
Note: Low-Rank Adapter algrithm for mindformers' pretrained model.
Reference: https://arxiv.org/abs/2106.09685
"""
import re

from mindspore import nn
from tk.delta.lora import LoRADense

from mindformers.modules.layers import Linear
from mindformers.models.llama.llama_layer import LlamaEmbedding
from mindformers.tools.logger import logger
from .pet_adapter import PetAdapter
from ..pet_config import PetConfig
from ..utils import re_match_list


def replace_embedding_head(net, config):
    """default replace all dense."""
    old_tok_embeddings = net._cells["tok_embeddings"]
    old_vocab_size = old_tok_embeddings.vocab_table_size
    new_vocab_size = config.expand_vocab_size
    dest_tok_embeddings = LlamaEmbedding(
        vocab_table_size=new_vocab_size,
        embedding_size=old_tok_embeddings.embedding_size,)
    # 加载原始llama embedding权重
    # dest_tok_embeddings.embedding_weight[:old_vocab_size] = old_tok_embeddings.embedding_weight
    # 分布式切片
    net._cells["tok_embeddings"] = dest_tok_embeddings

    # old_lm_head = lm_head
    # dest_lm_head = Linear(in_channels=old_lm_head.in_channels,
    #     out_channels=new_vocab_size,
    #     has_bias=old_lm_head.has_bias,
    #     compute_dtype=old_lm_head.dtype,
    #     transpose_b=old_lm_head.transpose_b)
    
    # if old_lm_head.has_bias:
    #     dest_lm_head.bias = old_lm_head.bias
    #     dest_lm_head.bias_add = old_lm_head.bias_add
    # 加载原始llama lm_head权重
    # if old_lm_head.transpose_b:
    #     dest_lm_head.weight[:, :old_vocab_size] = old_lm_head.weight
    #     dest_lm_head.bias[:, :old_vocab_size] = old_lm_head.bias
    # else:
    #     dest_lm_head.weight[:old_vocab_size] = old_lm_head.weight
    #     dest_lm_head.bias[:old_vocab_size] = old_lm_head.bias
    return net


class PretrainAdapter(PetAdapter):
    r"""
    LoraAdapter is the adapter to modify the pretrained model, which uses lora tuning algorithm.

    Args:
        model (BaseModel): The base pretrained model of mindformers.
        pet_config (PetConfig): The configurition of the Pet model.
    Return:
        model (BaseModel): The model replace the linear layer with lora dense layer.
    Examples:
        1.modify certain task of llama
        >>> from mindformers.pet.tuners.lora_adapter import LoraAdapter
        >>> class LlamaForCausalLMWithLora(LlamaForCausalLM):
        >>>        def __init__(self, config: LlamaConfig = None, pet=None):
        >>>            super().__init__(config)
        >>>            # get Pet tuning model.
        >>>            self.pet = pet
        >>>            self.pet.pet_config.reg_rules = r'.*wq|.*wv'
        >>>            self.model = LoraAdapter.get_pet_model(self.model, self.pet.pet_config)
        >>>            # freeze pretrained model
        >>>            PetAdapter.freeze_pretrained_model(self, self.pet.pet_type)
        2.modify certain model of llama
        >>> from mindformers.pet.tuners.lora_adapter import LoraAdapter
        >>> from mindformers.model.llama import LlamaModel
        >>> from mindformers.pet.pet_config import LoraConfig
        >>> llama_model = LlamaModel()
        >>> pet_config = LoraConfig()
        >>> llama_pet_model = LoraAdapter.get_pet_model(llama_model, pet_config)
    """
    @classmethod
    def get_pet_model(cls, model: nn.Cell = None, config: PetConfig = None):
        model = model if model else PetAdapter.get_pretrained_model(config)
        model = replace_embedding_head(model, config)
        return model
