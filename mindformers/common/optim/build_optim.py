# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Build Optimizer API."""
import inspect

from mindspore import nn

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


def build_optim(
        config: dict = None, default_args: dict = None,
        module_type: str = 'optimizer', class_name: str = None, **kwargs):
    """Build Optimizer API."""
    if config is None and class_name is None:
        return None
    if config is not None:
        return MindFormerRegister.get_instance_from_cfg(
            config, MindFormerModuleType.OPTIMIZER, default_args=default_args)
    return MindFormerRegister.get_instance(module_type, class_name, **kwargs)


def register_ms_optim():
    """ register MindSpore builtin optimizer class. """
    for module_name in dir(nn.optim):
        if module_name.startswith('__'):
            continue
        optim = getattr(nn.optim, module_name)
        if inspect.isclass(optim):
            MindFormerRegister.register_cls(
                optim, MindFormerModuleType.OPTIMIZER)


register_ms_optim()