# Neural Network Library (nn)

**Lua/Torch neural network framework for OpenCog Collection**

This is an implementation of the classic `torch/nn` library providing foundational neural network modules for the OpenCog Collection (OCC) monorepo. It serves as the base for the a9nn cognitive agent framework.

## Overview

The `nn` package provides:
- **Module**: Base class for all neural network components
- **Container**: Base class for modules containing other modules  
- **Criterion**: Base class for loss functions
- Standard neural network layers and activation functions
- Training utilities

## Installation

The nn library is part of the OpenCog Collection monorepo. It requires Lua and Torch to be installed.

```bash
# In the OCC repository root
luarocks install torch
```

## Quick Start

### Simple Multi-Layer Perceptron

```lua
require('nn')

-- Build a 3-layer MLP: 10 -> 25 -> 10 -> 1
local mlp = nn.Sequential()
mlp:add(nn.Linear(10, 25))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(25, 10))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(10, 1))

-- Create loss function
local criterion = nn.MSECriterion()

-- Prepare dataset (table of {input, target} pairs)
local dataset = {}
for i = 1, 1000 do
   local input = torch.randn(10)
   local target = torch.Tensor(1):fill(math.sin(input:sum()))
   dataset[i] = {input, target}
end

-- Train with stochastic gradient descent
local trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.01
trainer.maxIteration = 100
trainer:train(dataset)

-- Test the network
local testInput = torch.randn(10)
local prediction = mlp:forward(testInput)
print('Prediction:', prediction[1])
```

### Classification Example

```lua
require('nn')

-- Build classifier for 10 classes
local classifier = nn.Sequential()
classifier:add(nn.Linear(100, 50))
classifier:add(nn.ReLU())
classifier:add(nn.Linear(50, 10))
classifier:add(nn.LogSoftMax())

-- Use negative log-likelihood loss
local criterion = nn.ClassNLLCriterion()

-- Or use cross-entropy (combines LogSoftMax + NLL)
-- local classifier = nn.Sequential()
-- classifier:add(nn.Linear(100, 50))
-- classifier:add(nn.ReLU())
-- classifier:add(nn.Linear(50, 10))
-- local criterion = nn.CrossEntropyCriterion()

-- Training loop
for epoch = 1, 10 do
   for i = 1, #dataset do
      local input = dataset[i][1]
      local target = dataset[i][2] -- class index (1-10)
      
      -- Forward
      local output = classifier:forward(input)
      local loss = criterion:forward(output, target)
      
      -- Backward
      classifier:zeroGradParameters()
      local gradOutput = criterion:backward(output, target)
      classifier:backward(input, gradOutput)
      
      -- Update
      classifier:updateParameters(0.01)
   end
end
```

## Core Modules

### Base Classes

- **nn.Module**: Base class for all layers
- **nn.Container**: Base for containers holding multiple modules
- **nn.Criterion**: Base class for loss functions

### Layers

- **nn.Linear(inputSize, outputSize)**: Fully connected layer
  ```lua
  layer = nn.Linear(10, 5)  -- 10 inputs, 5 outputs
  ```

### Activation Functions

- **nn.Tanh()**: Hyperbolic tangent
- **nn.Sigmoid()**: Sigmoid function
- **nn.ReLU()**: Rectified Linear Unit
- **nn.SoftMax()**: SoftMax normalization
- **nn.LogSoftMax()**: Log of SoftMax

### Containers

- **nn.Sequential()**: Chain modules sequentially
  ```lua
  net = nn.Sequential()
  net:add(nn.Linear(10, 5))
  net:add(nn.Tanh())
  ```

- **nn.Parallel(inputDim, outputDim)**: Apply modules to input slices
- **nn.Concat(dimension)**: Concatenate module outputs

### Loss Functions (Criterions)

- **nn.MSECriterion()**: Mean Squared Error
- **nn.ClassNLLCriterion()**: Negative Log Likelihood (for classification)
- **nn.CrossEntropyCriterion()**: Cross Entropy (LogSoftMax + NLL)

### Training

- **nn.StochasticGradient(module, criterion)**: Simple SGD trainer
  ```lua
  trainer = nn.StochasticGradient(model, criterion)
  trainer.learningRate = 0.01
  trainer.maxIteration = 100
  trainer:train(dataset)
  ```

## Module API

All modules inherit from `nn.Module` and implement:

```lua
-- Forward pass
output = module:forward(input)

-- Backward pass  
gradInput = module:backward(input, gradOutput)

-- Or separately:
gradInput = module:updateGradInput(input, gradOutput)
module:accGradParameters(input, gradOutput, scale)

-- Parameter management
params, gradParams = module:parameters()
module:zeroGradParameters()
module:updateParameters(learningRate)

-- Mode switching
module:training()   -- Enable training mode
module:evaluate()   -- Enable evaluation mode

-- Type conversion
module:float()      -- Convert to float
module:double()     -- Convert to double
module:cuda()       -- Convert to CUDA tensors

-- Serialization
clone = module:clone()
```

## Criterion API

All criterions inherit from `nn.Criterion`:

```lua
-- Compute loss
loss = criterion:forward(input, target)

-- Compute gradient
gradInput = criterion:backward(input, target)
```

## Integration with OCC

This nn library integrates with the OpenCog Collection:

1. **AtomSpace Integration**: Modules can store/load parameters in AtomSpace
2. **Cognitive Agent Framework**: Base for a9nn cognitive agents
3. **Build System**: Integrated with OCC CMake build
4. **Documentation**: Part of OCC documentation system

## Advanced Usage

### Custom Modules

Create custom layers by inheriting from `nn.Module`:

```lua
local MyLayer, parent = torch.class('nn.MyLayer', 'nn.Module')

function MyLayer:__init(inputSize, outputSize)
   parent.__init(self)
   self.weight = torch.Tensor(outputSize, inputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self:reset()
end

function MyLayer:reset()
   self.weight:uniform(-0.1, 0.1)
end

function MyLayer:updateOutput(input)
   -- Implement forward pass
   self.output = torch.mv(self.weight, input)
   return self.output
end

function MyLayer:updateGradInput(input, gradOutput)
   -- Implement backward pass
   self.gradInput = torch.mv(self.weight:t(), gradOutput)
   return self.gradInput
end

function MyLayer:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   self.gradWeight:addr(scale, gradOutput, input)
end

function MyLayer:parameters()
   return {self.weight}, {self.gradWeight}
end
```

### Module Inspection

```lua
-- Print module structure
print(model)

-- Get all parameters
params, gradParams = model:getParameters()
print('Total parameters:', params:nElement())

-- Access submodules
for i = 1, model:size() do
   print('Module', i, ':', model:get(i))
end
```

## Testing

Run the test suite:

```bash
cd nn/tests
lua test_modules.lua
```

## Documentation

See the `doc/` directory for detailed documentation:
- `module.md`: Module interface reference
- `containers.md`: Container classes
- `criterions.md`: Loss functions
- `training.md`: Training procedures

## Related Components

- **a9nn**: Cognitive agent framework extending nn
- **nnecco-a9nn**: NNECCO cognitive architecture
- **AtomSpace**: Hypergraph knowledge representation

## References

- [torch/nn](https://github.com/torch/nn): Original Torch neural network library
- [OpenCog Collection](https://github.com/opencog/occ): OCC monorepo
- [Torch Documentation](http://torch.ch/docs/): Lua/Torch framework

## License

BSD License (inherited from torch/nn)

Part of the OpenCog Collection (OCC) monorepo.
