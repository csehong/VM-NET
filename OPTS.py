#!/usr/bin/python

class OPTS:
  def __init__(self, name):
    self.__name = name
    
  def get_keys(self):
    return self.__dict__.keys()
  
  def assert_all_keys_valid(self):
    for k in self.get_keys():
      try:
        assert self.__dict__[k] is not None
      except AssertionError:
        print('%s: Option "%s" is not valid'%(self.__name, k))
        exit()
    