#!/usr/bin/python

import os.path
from datetime import datetime

class Logger:
  def __init__(self, path):
    if path.endswith('.txt'):
      file_path = path
    else:
      file_path = os.path.join(path,'log.txt')
    self.log_file = open(file_path,'a')
  
  def write(self, str):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_format = '[%s] %s'%(current_time, str)
    self.log_file.write(log_format + '\n')
    print (log_format)
  
  def __del__(self):
    self.log_file.close()