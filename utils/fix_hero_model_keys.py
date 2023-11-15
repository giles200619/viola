#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append('./')
sys.path.append('./simplerecon')
import torch
os.chdir('./simplerecon')
print(os.getcwd())
import options
#add missing keys
ckpt = torch.load('./weights/hero_model.ckpt')
print('original keys:',ckpt.keys())
ckpt['pytorch-lightning_version'] = '0.0.0'
ckpt['global_step'] = '0'
ckpt['epoch'] = '0'
print('added keys:',ckpt.keys())
torch.save(ckpt, './weights/hero_model.ckpt')