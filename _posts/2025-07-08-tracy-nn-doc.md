---
layout: post
title:  "Tracy NN Documentation"
date:   2025-06-07 19:10:00 +0330
categories: [update]
---

# What is Tracy NN:


Tracy NN is a debugger tool for better understanding the matrix operations in
pytorch deep learning library. Honestly I'm bad at Linear Algebra and it gets
very difficult & frustrating to follow complicated architectures like 
transformers and truly internalize the events and operation that are happening
in there.

So while working on my own project i came up with the idea and really used the 
help of pytorch hook functions to log the Tensor shapes being transormed by the
operation or nn.Modules.

I hope it becomes a useful tool for everyone to better understand deep learning.

# How to install it:

```bash
pip install tracy_nn
```

# How to use it:


```python
import torch
import torch.nn as nn
from tracy_nn import Tracer

# Make a dummy input tensor (or not dummy. you decide!)
x = torch.rand(batch_size, seq_len, d_in) 

# Create an instance of your Module
mha = MHA(d_in, d_out, seq_len, num_heads, context_window, dropout)

# Create an instance of tracer and name it (use the name of what's being traced)
tracer = Tracer('MHA')

# Start the tracing process
tracer.start(mha)       

# One forward pass is enough for the tracing process
output = mha(x)         

# Stop the tracing process
tracer.stop()           
```

Tracy NN even provides easier ways:

```python
from tracy_nn import Tracer
with tracer.trace(model):
    output = model(input_tensor)
```

or:

```python
from tracy_nn import trace_model
tracer = trace_model(model, 'My Model')
with tracer.trace(model):
    output = model(input_tensor)
```

# Result:


```bash
  ● TRACING MHA STARTED
  │
  ● query_weights (Linear) (in_features: 6, out_features: 6)
  ├─ Input: Tensor [10, 20, 6] @ cpu
  └─ Output: Tensor [10, 20, 6] @ cpu
  │
  ● key_weights (Linear) (in_features: 6, out_features: 6)
  ├─ Input: Tensor [10, 20, 6] @ cpu
  └─ Output: Tensor [10, 20, 6] @ cpu
  │
  ● value_weights (Linear) (in_features: 6, out_features: 6)
  ├─ Input: Tensor [10, 20, 6] @ cpu
  └─ Output: Tensor [10, 20, 6] @ cpu
  │
  ● Tensor.view(10, 20, 3, 2) @ [MHA.split_heads]
  ├─ Input: Tensor [10, 20, 6] @ cpu
  └─ Output: Tensor [10, 20, 3, 2] @ cpu
  │
  [...]
  │
  ● Tensor.@ @ [MHA.calculate_attention]
  ├─ Left: Tensor [10, 3, 20, 20] @ cpu
  ├─ Right: Tensor [10, 3, 20, 2] @ cpu
  └─ Output: Tensor [10, 3, 20, 2] @ cpu
  │
  ● Tensor.transpose(1, 2) @ [MHA.combine_heads]
  ├─ Input: Tensor [10, 3, 20, 2] @ cpu
  └─ Output: Tensor [10, 20, 3, 2] @ cpu
  │
  ● Tensor.view(10, 20, 6) @ [MHA.combine_heads]
  ├─ Input: Tensor [10, 20, 3, 2] @ cpu
  └─ Output: Tensor [10, 20, 6] @ cpu
  │
  ● out_projection (Linear) (in_features: 6, out_features: 6)
  ├─ Input: Tensor [10, 20, 6] @ cpu
  └─ Output: Tensor [10, 20, 6] @ cpu
  │
  ● TRACING MHA COMPLETE
```

# Note:

- Don't use tracy nn while training, logs will fill your terminal and slow the 
training process, tracy is meant to be used at designing & debugging process.
that's why only one forward pass will be enough.

- Some operations like torch.cat() don't get logged so better to use the
equivalent methods like Tensor.cat(), but don't worry most operations, even
python standard operations like @, and / are coverd by translation to their
torch equivalent.

