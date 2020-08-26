# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:15:02 2019

@author: Lenovo
"""


import re

def return_patient_ID(comp_ID):
     ID_match = re.search(r'(?<=TCGA-[A-Z0-9][A-Z0-9]-)[0-9A-Z]*', comp_ID, re.I)
     return ID_match.group()
