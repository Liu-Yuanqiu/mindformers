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
'''download_tools'''
import time
import os
import requests
import urllib3
from tqdm import tqdm

from .logger import logger
class StatusCode:
    '''StatusCode'''
    succeed = 200


def downlond_with_progress_bar(url, filepath, chunk_size=1024, timeout=4):
    '''downlond_with_progress_bar'''
    if not os.path.exists(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    start = time.time()

    try:
        response = requests.get(url, stream=True, timeout=timeout)
    except (TimeoutError, urllib3.exceptions.MaxRetryError,
            requests.exceptions.ProxyError,
            requests.exceptions.ConnectionError):
        logger.error("Connect error, please download %s to %s.", url, filepath)
        return False

    size = 0
    content_size = int(response.headers['content-length'])
    if response.status_code == StatusCode.succeed:
        logger.info('Start download %s,[File size]:{%.2f} MB',
                    filepath, content_size / chunk_size /1024)
        with open(filepath, 'wb') as file:
            for data in tqdm(response.iter_content(chunk_size=chunk_size)):
                file.write(data)
                size += len(data)
        file.close()
        end = time.time()
        logger.info('Download completed!,times: %.2fs', (end - start))
        return True

    logger.error("%s is unconnected!", url)
    return False
