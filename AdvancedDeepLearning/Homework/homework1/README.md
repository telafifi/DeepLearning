# Homework 1 - Low memory training and inference

In this homework, we will start with a large deep network `BigNet`. To keep things simple we keep it at 73MB.

First, familiarize yourself with `BigNet` (see `homework/bignet.py`). You should see:

- Layer definitions
- The main network `BigNet`
- A `load` function

If you're curious to see how big `BigNet` actually is call

```bash
python3 -m homework.stats bignet
```

It should show this (up to rounding errors depending on your PyTorch backend)

```
 bignet
Trainable params          18.90 M
Non-trainable params       0.00 M
Total params              18.90 M
Theoretical memory        72.11 MB
Actual memory             72.11 MB
Forward memory             8.12 MB
Backward memory           80.23 MB
```

In this homework, you will implement four additional versions of `BigNet` using various memory-saving techniques:

- Half precision networks (in `homework/half_precision.py`)
- LoRA with a half-precision base network (in `homework/lora.py`)
- 4-bit quantized networks (in `homework/low_precision.py`)
- QLoRA (in `homework/qlora.py`)

For each, you should first define any net layers you might need. It is fine to use PyTorch's built-in layers, but you should not use any external dependencies such as hugging face or torchtune.

## Grading Criteria

We will grade your solution using two criteria: Forward only (for all models), Forward-backward (LoRA and QLoRA).

### Forward only (80 pts)

For each implementation, we will benchmark the total memory used and verify that outputs from your implementation match the original `BigNet` (up to some error tolerance). Each implementation scores up to 20 pts.

### Forward-backward (20 pts)

We will train your LoRA and QLoRA implementation for a few steps to verify that they can fit a certain training objective. Each implementation scores up to 10 pts.

## Half precision (20 pts)

Let's start with `homework/half_precision.py`. Implement a `BigNet` whose weights are stored in half-precision.
There are at least a dozen different ways to implement this in PyTorch. The starter code nudges you towards a very particular implementation: Redefine a BigNet and replace all `torch.nn.Linear` layers with your implementation of a `HalfLinear` layer.

This takes advantage of several properties of `torch.nn.Modules`. First, if the parameters of this `HalfLinear` layers have the same name as `torch.nn.Linear` then loading and storing parameters from a checkpoint will work out of the box (even if the dtype slightly mismatches `float16` vs `float32`).
The easiest way to implement `HalfLinear` is to directly inherit from `torch.nn.Linear` and use the right parameters in `super().__init__`.

You have to override the `forward` function of your new `HalfLinear` layer to cast tensor types to match (PyTorch does not like to mix `float16` and `float32` in common operations).

When you're done test your new network:

```bash
python3 -m homework.stats bignet half_precision
```

You should see a 50% decrease in memory use throughout:

```
 bignet     half_precision
Trainable params          18.90 M         0.01 M
Non-trainable params       0.00 M        18.89 M
Total params              18.90 M        18.90 M
Theoretical memory        72.11 MB       36.07 MB
Actual memory             72.11 MB       36.07 MB
Forward memory             8.12 MB        0.00 MB
Backward memory           80.23 MB        0.04 MB
```

`Trainable params` and `Backward memory` are near zero in our implementation. We disabled back-propagation through the float16 linear layer. It is numerically not very stable.

You can also compare the outputs of the two models:

```bash
python3 -m homework.compare bignet half_precision
```

They should match within a tolerance of 0.002.

## LoRA (30 pts)

Let's start with `homework/lora.py`. We will build on the `HalfBigNet` here, and add a LoRA adapter to all linear layers. There are again several ways to implement this. The easiest is to again modify the network, especially the linear layers, but keep the overall names of layers the same. That way PyTorch will again load all weights for us.

Start by implementing `LoRALinear`. It is fine to inherit from `HalfLinear` and implement the LoRA adapter on top. Make sure to keep the LoRA linear layers at `float32` precision, otherwise they might not train well.

Use `stats` and `compare` to test your implementation.

```bash
python3 -m homework.stats bignet lora
python3 -m homework.compare bignet lora
```

Finally, let's see if your LoRA adapter trains:

```bash
python3 -m homework.fit lora
```

This script fits the model to 1000 samples of random noise half with a label 0, half with a label 1. Since the input dimension is large (1024) and the number of samples is small, this will almost certainly overfit within very few iterations (about 30).

## 4-Bit Quantization (20 pts)

Let's start with `homework/low_precision.py`. Here, we implement 4-bit quantization for weights of all linear layers. We provide you with a very basic PyTorch native 4-bit block quantizer.
`block_quantize_4bit` takes a 1D `torch.Tensor` and quantizes groups of `group_size` values into 4 bits each.
The quantization works as follows:

- Given a group of values $v_1 \ldots v_k$
- Find the largest absolute value $\hat v = \max_i |v_i|$
- Store $\hat v$ in `float16` (using 16 / k bits per value)
- Store $(v_i + \hat v) / (2*\hat v)$ as a value from 0 to 15 (using 4-bits per value)
- Finally, pack two consecutive 4-bit values into a single byte

The output is two tensors `x_quant_4` quantizes values in `int8` format, and `normalization` in `float16` format.

`block_dequantize_4bit` reverses this process.

Since PyTorch does not natively support this sort of quantization, the corresponding `Linear4Bit` layer requires some plumbing:

- First, the `weight_q4` and `weight_norm` parameters need to be manually constructed and registered. This gaurantees any `Module.to(device)` function moves the quantized weights on the proper device.
- Next, the weights in a checkpoint are stored in the `...weight` `state_dict`, but the layer does not have a `weight` parameter. This requires us to override the `load_state_dict` function, which is done through a `_load_state_dict_pre_hook`. A large part of this is implemented in the starter code, including obtaining the weights from the `state_dict` and some bookkeeping. You should implement quantization in the `_load_state_dict_pre_hook`.
- Finally, the forward function needs to dequantize weights before they can be used in `torch.nn.functional.linear`.

If everything goes well, you should see some massive savings in memory:

```bash
python3 -m homework.stats bignet low_precision
```

This should lead to an almost 7x reduction in memory. Think about why not 8x (float32: 32 bits -> 4 bits)?

```
 bignet     low_precision
Trainable params          18.90 M         0.03 M
Non-trainable params       0.00 M        10.62 M
Total params              18.90 M        10.65 M
Theoretical memory        72.11 MB       11.36 MB
Actual memory             72.11 MB       11.36 MB
Forward memory             8.12 MB        0.00 MB
Backward memory           80.23 MB        0.04 MB
```

## Q-LoRA (30 pts)

Let's start with `homework/qlora.py`. If you inherit from `Linear4Bit`, `QLoRA` should not be much harder than `LoRA`.

## Grading

The test grader we provide

```bash
python3 -m grader homework -v
```

This will run a subset of test cases we use during the actual testing.
The point distributions will be the same, but we will use additional test cases.
The performance of the test grader may vary.

## Extra Credit (5 pts)

Can you compress the model below 4 bits per parameters?
Let's start with `homework/lower_precision.py`. The memory requirements here are very strict (<9MB), with still a decent accuracy.
This will require implementing your own block quantizer. Only attempt this if you have plenty of free time!

## Submission

Once you finished the assignment, create a submission bundle using:

```bash
python3 bundle.py homework [YOUR UT ID]
```

Submit the zip file on Canvas. Please note that the maximum file size our grader accepts is **20MB**. Please keep your solution compact.
Please double-check that your zip file was properly created, by grading it again:

```bash
python3 -m grader [YOUR UT ID].zip
```

## Online grader

We will use an automated grader through Canvas to grade all your submissions. There is a soft limit of **5** submissions per assignment. Please contact the course staff before going over this limit, otherwise your submission might be counted as invalid.

The online grading system will use a slightly modified version of Python and the grader:

- Please do not use the `exit` or `sys.exit` command, it will likely lead to a crash in the grader
- Please do not try to access, read, or write files outside the ones specified in the assignment. This again will lead to a crash. File writing is disabled.
- Network access is disabled. Please do not try to communicate with the outside world.
- Forking is not allowed!
- `print` or `sys.stdout.write` statements from your code are ignored and not returned.

Please do not try to break or hack the grader. Doing so will have negative consequences for your standing in this class and the program.

## Installation

We encourage using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to install the required packages.

```bash
conda create --name advances_in_deeplearning python=3.12 -y
conda activate advances_in_deeplearning
```

First, install [PyTorch](https://pytorch.org/get-started/locally/)

Then install additional dependencies:

```bash
pip install -r requirements.txt
```
