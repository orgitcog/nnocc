# nn Module Documentation

## Overview

The nn (neural network) module provides a Lua/Torch-based neural network framework for the OpenCog Collection. It implements the classic torch/nn API with extensions for cognitive agent architecture.

## Module Interface

All neural network modules in nn inherit from `nn.Module`, which provides a standard interface for forward and backward propagation.

### Core Methods

#### forward(input)
Performs the forward pass, computing the output from the input.

```lua
output = module:forward(input)
```

Equivalent to calling `updateOutput(input)` directly.

#### backward(input, gradOutput, scale)
Performs the backward pass, computing gradients with respect to input and parameters.

```lua
gradInput = module:backward(input, gradOutput, scale)
```

This is equivalent to:
```lua
gradInput = module:updateGradInput(input, gradOutput)
module:accGradParameters(input, gradOutput, scale)
```

#### updateOutput(input)
Computes the module's output given an input. Must be implemented by subclasses.

#### updateGradInput(input, gradOutput)
Computes the gradient with respect to the input, given the gradient with respect to the output.

#### accGradParameters(input, gradOutput, scale)
Accumulates gradients with respect to the module's parameters. Only relevant for modules with learnable parameters.

### Parameter Management

#### parameters()
Returns two tables: one containing the module's parameters, and one containing the corresponding gradient parameters.

```lua
params, gradParams = module:parameters()
```

#### zeroGradParameters()
Sets all parameter gradients to zero.

```lua
module:zeroGradParameters()
```

#### updateParameters(learningRate)
Updates parameters using their gradients:

```lua
module:updateParameters(0.01)  -- learning rate = 0.01
```

This implements the basic SGD update:
```
parameter = parameter - learningRate * gradient
```

#### getParameters()
Returns all parameters and gradients as flat vectors.

```lua
flatParams, flatGradParams = module:getParameters()
```

### Type Conversion

#### type(tensorType)
Converts the module and all its tensors to a different type.

```lua
module:type('torch.FloatTensor')
module:type('torch.DoubleTensor')
module:type('torch.CudaTensor')
```

Convenience methods:
```lua
module:float()   -- Convert to float
module:double()  -- Convert to double  
module:cuda()    -- Convert to CUDA
```

### Training/Evaluation Mode

#### training()
Sets the module to training mode. Some modules behave differently during training vs evaluation (e.g., Dropout, BatchNorm).

```lua
module:training()
```

#### evaluate()
Sets the module to evaluation mode.

```lua
module:evaluate()
```

### Cloning

#### clone()
Creates a deep copy of the module.

```lua
moduleCopy = module:clone()
```

### Inspection

#### __tostring__()
Returns a string representation of the module.

```lua
print(module)  -- Calls __tostring__
```

## Available Modules

### Linear Layers

**nn.Linear(inputSize, outputSize, [bias])**

Fully connected linear transformation: `y = Wx + b`

- `inputSize`: Size of input features
- `outputSize`: Size of output features  
- `bias`: Whether to include bias term (default: true)

Parameters:
- `weight`: Matrix of size (outputSize × inputSize)
- `bias`: Vector of size (outputSize)

### Activation Functions

**nn.Tanh()**

Applies hyperbolic tangent element-wise: `y = tanh(x)`

**nn.Sigmoid()**

Applies sigmoid element-wise: `y = 1/(1+exp(-x))`

**nn.ReLU([inplace])**

Rectified Linear Unit: `y = max(0, x)`

- `inplace`: Whether to modify input in-place (default: false)

**nn.SoftMax()**

Applies softmax normalization: `y_i = exp(x_i) / sum_j exp(x_j)`

**nn.LogSoftMax()**

Applies log-softmax: `y_i = x_i - log(sum_j exp(x_j))`

## Container Modules

**nn.Sequential()**

Chains modules sequentially. Output of one module becomes input to the next.

```lua
model = nn.Sequential()
model:add(nn.Linear(10, 5))
model:add(nn.Tanh())
model:add(nn.Linear(5, 1))
```

Methods:
- `add(module)`: Adds a module to the end
- `get(index)`: Gets module at index
- `size()`: Returns number of modules

**nn.Parallel(inputDimension, outputDimension)**

Applies each child module to the corresponding slice of input, concatenates outputs.

**nn.Concat(dimension)**

Applies each child module to the same input, concatenates outputs along specified dimension.

## Criterion Modules

Criterions compute loss and its gradient with respect to the network output.

### Methods

#### forward(input, target)
Computes the loss value.

```lua
loss = criterion:forward(output, target)
```

#### backward(input, target)
Computes the gradient of the loss with respect to the input.

```lua
gradInput = criterion:backward(output, target)
```

### Available Criterions

**nn.MSECriterion()**

Mean Squared Error loss:
```
loss = (1/n) * sum((input - target)^2)
```

**nn.ClassNLLCriterion([weights])**

Negative Log Likelihood loss for classification. Use with nn.LogSoftMax.

```lua
criterion = nn.ClassNLLCriterion()
loss = criterion:forward(logProbs, targetClass)
```

- `weights`: Optional tensor of class weights
- `targetClass`: Integer class index (1-based)

**nn.CrossEntropyCriterion([weights])**

Cross-entropy loss. Combines LogSoftMax + ClassNLLCriterion.

```lua
criterion = nn.CrossEntropyCriterion()
loss = criterion:forward(logits, targetClass)
```

## Training

**nn.StochasticGradient(module, criterion)**

Simple SGD trainer for neural networks.

```lua
trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = 0.01
trainer.learningRateDecay = 0  
trainer.maxIteration = 100
trainer.shuffleIndices = true
trainer.verbose = true
trainer:train(dataset)
```

Dataset format: Table of `{input, target}` pairs.

Properties:
- `learningRate`: Learning rate (default: 0.01)
- `learningRateDecay`: Decay factor (default: 0)
- `maxIteration`: Maximum epochs (default: 25)
- `shuffleIndices`: Shuffle samples (default: true)
- `verbose`: Print progress (default: true)

## Example: Complete Training Loop

```lua
require('nn')

-- Build model
local model = nn.Sequential()
model:add(nn.Linear(784, 128))
model:add(nn.ReLU())
model:add(nn.Linear(128, 10))
model:add(nn.LogSoftMax())

-- Create criterion
local criterion = nn.ClassNLLCriterion()

-- Prepare dataset
local dataset = {}
for i = 1, 1000 do
   local input = torch.randn(784)
   local target = math.random(1, 10)
   dataset[i] = {input, target}
end

-- Manual training loop
local learningRate = 0.01
for epoch = 1, 10 do
   local totalLoss = 0
   
   for i = 1, #dataset do
      local input = dataset[i][1]
      local target = dataset[i][2]
      
      -- Forward pass
      local output = model:forward(input)
      local loss = criterion:forward(output, target)
      totalLoss = totalLoss + loss
      
      -- Backward pass
      model:zeroGradParameters()
      local gradOutput = criterion:backward(output, target)
      model:backward(input, gradOutput)
      
      -- Update parameters
      model:updateParameters(learningRate)
   end
   
   print(string.format('Epoch %d: loss = %.4f', epoch, totalLoss / #dataset))
end

-- Or use StochasticGradient
local trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = 0.01
trainer.maxIteration = 10
trainer:train(dataset)
```

## See Also

- [README.md](../README.md): Quick start guide
- [containers.md](containers.md): Container classes
- [criterions.md](criterions.md): Loss functions  
- [training.md](training.md): Training procedures
