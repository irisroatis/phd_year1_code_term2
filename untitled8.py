#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:50:15 2023

@author: roatisiris
"""

import pycaret
import pandas as pd
import pycaret.classification 

from pycaret.datasets import get_data
data = get_data('diabetes')

exp = pycaret.classification.ClassificationExperiment()
type(exp)

exp.setup(data, target = 'Class variable')