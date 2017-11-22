# -*- coding: utf-8 -*-
"""
File manager

@author: Chien-Chih Lin
"""

from datetime import datetime
import os
import shutil

def _clean_files(rm_path, bak_path):
    """
    - make dirs for backup
    - clean backup dir
    - move dirs inside backup dir
    - make dirs for program to output files
    """
    if not os.path.isdir(rm_path):
        os.makedirs(rm_path)
    if not os.path.isdir(bak_path):
        os.makedirs(bak_path)
    
    shutil.rmtree(bak_path)
    shutil.move(rm_path, bak_path)

    os.makedirs(rm_path)

    return None

def backup_files(dir_list):
    for directory in dir_list:
        _clean_files(directory, os.path.join("backup", directory))
    return None

