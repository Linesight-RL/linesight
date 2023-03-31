# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:04:27 2023

@author: chopi
"""

def foo():
    import sys
    variable = 12341324
    print(sys.getrefcount(variable))
    print(hex(id(variable)))

foo()