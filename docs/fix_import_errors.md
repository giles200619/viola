## Fixing import errors

If you run into import errors in `ms_deform_attn.py` with the package `MultiScaleDeformableAttention-1.0`,<br />
modify the file: `<path to your Anaconda>/envs/viola/lib/python3.9/site-packages/MultiScaleDeformableAttention-1.0-py3.9-linux-x86_64.egg/modules/ms_deform_attn.py`:<br />
from:
```
from ..functions import MSDeformAttnFunction
from ..functions.ms_deform_attn_func import ms_deform_attn_core_pytorch
```
to:
```
import sys
sys.path.append('..')
from functions import MSDeformAttnFunction
from functions.ms_deform_attn_func import ms_deform_attn_core_pytorch
```