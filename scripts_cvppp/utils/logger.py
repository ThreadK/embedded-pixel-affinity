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

        # log_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),"logs","Running_logs") #存放打印的日志的目录
        log_path = os.path.join('./logs', 'Running_logs')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        timestr = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
        self.log_name = os.path.join(log_path,timestr+".log")  #log's name includes time
        self.logger.setLevel(logging.INFO)
        self.formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s]: %(message)s')  # [2019-05-15 14:48:52,947] - test.py] - ERROR: this is error
        fh = logging.FileHandler(self.log_name, 'a', encoding='utf-8')
        # 设置日志等级
        fh.setLevel(logging.INFO)
        # 设置handler的格式对象
        fh.setFormatter(self.formatte