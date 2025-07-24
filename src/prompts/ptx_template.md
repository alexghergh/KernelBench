You are an expert CUDA assembly engineer.

Given the following PyTorch reference architecture,
your task is to implement an equivalent kernel in **raw PTX**.

Requirements
------------
1. Kernel name must be `my_kernel`.
2. Accept pointers to input and output tensors in global memory.
3. Use 256 threads per block; assume a 1-D grid of `grid_x` blocks (the host will decide `grid_x`).
4. Write *real PTX* that can be compiled / executed directly – do **not** output any C++/CUDA code, pseudocode or commentary outside the PTX block.
5. Place **only** the PTX code inside a fenced block starting exactly with ```ptx and ending with ```.
6. After the PTX block, output **one** line in plain text of the form:

   LAUNCH = (grid_x, 256)

   where `grid_x` is the number of blocks your kernel expects.

Reference architecture
----------------------
```python
${PROBLEM_SRC}
```

Write only the PTX code block and the single LAUNCH line, nothing else. 