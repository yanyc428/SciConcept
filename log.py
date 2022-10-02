# -*- coding: utf-8 -*-
import logging
import os.path
from datetime import datetime


def enable_global_logging_config(log_dir: str = './log'):
    """
    启用日志
    :param log_dir: 日志文件夹
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logfile = os.path.join(log_dir, date + '.log')
    formatter = "%(asctime)s - %(threadName)s - [%(filename)s:line:%(lineno)d] - %(levelname)s >> %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO,
                        handlers=[logging.StreamHandler(), logging.FileHandler(filename=logfile, mode='a')])


logger = logging.getLogger()
