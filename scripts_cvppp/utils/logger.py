# -*- coding: utf-8 -*-
# @Time : 2021/3/30
# @Author : Xiaoyu Liu
# @Email : liuxyu@mail.ustc.edu.cn
# @Software: PyCharm

import logging
import time
import os

class Log():
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # log_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),"logs","Running