﻿# The MIT License (MIT)
#
# Copyright (c) 2016 Ilhan Polat
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""
:mod:`harold` is a Python package that provides tools for analyzing feedback
control systems and designing controllers.
"""
#from ._version import version as __version__
from ._global_constants import *
from ._aux_linalg import *
from ._polynomial_ops import *
from ._classes import *
from ._system_funcs import *
from ._bd_algebra import *
from ._solvers import *
from ._discrete_funcs import *
from ._frequency_domain import *
from ._system_props import *
from ._kalman_ops import *
from ._static_ctrl_design import *
from ._frequency_domain_plots import *
from ._time_domain import *
from ._time_domain_plots import *
