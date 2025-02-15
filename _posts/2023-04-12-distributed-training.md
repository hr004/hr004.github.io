---
title: Distributed Training
author: Hafizur Rahman
categories: [Computing]
tags: [Reinforcement Learning]
math: true
---


### Data Parallelism
Data Parallelism works by replicating the model on each device and updating the model parameters synchronously. Each device gets a different mini-batch of data and computes the gradients independently. The gradients are then averaged across all devices and the model parameters are updated. This is the most common way to train deep learning models in a distributed setting.

The equation for synchronizing the model parameters is given by:

$$
\begin{equation}
\theta_{t+1} = \theta_t - \alpha \frac{1}{n} \sum_{i=1}^{n} \nabla L(f(x_i; \theta_t), y_i)
\end{equation}
$$

where $\theta_t$ is the model parameters at time $t$, $\alpha$ is the learning rate, $n$ is the number of devices, $L$ is the loss function, $f$ is the model, $x_i$ is the input data, and $y_i$ is the target data.

### Model Parallelism

Model Parallelism works by splitting the model across multiple devices. Each device is responsible for computing a part of the model and the gradients are communicated between devices. This is useful when the model is too large to fit on a single device.
When to use Model Parallelism:
- When the model is too large to fit on a single device
- When the model has multiple components that can be computed independently in parallel.

How does it work?
- Split the model into multiple parts
- Assign each part to a different device
- Compute the gradients for each part independently
- Communicate the gradients between devices
- Update the model parameters

We can use APIs such as `rpc` to leverage the execution across different processes. Pytorch provides a `torch.distributed.rpc` module that allows us to define remote functions and call them across different processes.


### Benchmarking Distributed Training in Compute Canada

Let's take a look at how we can benchmark distributed training in Compute Canada. We will use PyTorch and Horovod to train a deep learning model on multiple GPUs.

here's the code snippet to train a deep learning model using Horovod:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import horovod.torch as hvd

# Initialize Horovod
hvd.init()

# Define the model
model = nn.Sequential(
	nn.Linear(784, 512),
	nn.ReLU(),
	nn.Linear(512, 10)
)

# Wrap the model with Horovod
model = hvd.DistributedDataParallel(model)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
for epoch in range(10):
	for data, target in train_loader:
		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
```

# References

