from cProfile import label
import numpy as np
import pandas as pd
import os
from utils import util


param = util.parse_arguments()
util.set_seeds(param.seed)