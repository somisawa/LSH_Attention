# A simple implementation of the LSH Attention in Reformer
## Description

Calculate Softmax layer of Attention in $O(L\log L)(L=sequence length)$ instead of $O(L^2)$ using the cross-polytope [Locality-Sensitive Hashing](https://arxiv.org/abs/1802.05751 ). For more detail, look at this [paper](https://arxiv.org/abs/2001.04451 ).



## Usage 

You only need `numpy >=1.18 `.

For example, 

```python
import numpy as np

from functions import normal_softmax, lsh_softmax

R = np.random.randn(100, 10000)
normal_sm = normal_softmax(R)
lsh_sm = lsh_softmax(R)
```



## Test

### Small size

Note: For better visibility, the diagonal components are rewritten to 0

<img width="847" alt="test" src="https://user-images.githubusercontent.com/37485236/79003287-3f403880-7b8d-11ea-97bc-9d3c6fc72a7b.png">

## Time complexity analysis

The execution times are plotted for sequence lengths of $2^i$($i=4, 5, \cdots , 15$).

![time_analysis_log](https://user-images.githubusercontent.com/37485236/79003750-2c7a3380-7b8e-11ea-9cf7-337ad0bb5413.png)
