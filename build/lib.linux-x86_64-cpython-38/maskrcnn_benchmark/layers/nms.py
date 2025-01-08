# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
from maskrcnn_benchmark import _C



# Only valid with fp32 inputs - give AMP the hint
nms = _C.nms

# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
