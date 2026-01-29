--[[
   OpenCog Neural Network Library (nn)
   
   Lua/Torch neural network framework implementing the classic torch/nn API
   with extensions for cognitive agent architecture.
   
   Based on: https://github.com/torch/nn
   
   Core Components:
   - Module: Base class for all neural network components
   - Container: Base class for modules containing other modules
   - Criterion: Base class for loss functions
   
   This is part of the OpenCog Collection (OCC) monorepo.
]]--

require('torch')

nn = {}

-- Version information
nn.version = '1.0.0'

-- Module management
nn.modules = {}

-- Load core components
require('nn.Module')
require('nn.Container')
require('nn.Criterion')

-- Load basic modules
require('nn.Linear')
require('nn.Tanh')
require('nn.Sigmoid')
require('nn.ReLU')
require('nn.LogSoftMax')
require('nn.SoftMax')

-- Load containers
require('nn.Sequential')
require('nn.Parallel')
require('nn.Concat')

-- Load criterions
require('nn.MSECriterion')
require('nn.ClassNLLCriterion')
require('nn.CrossEntropyCriterion')

-- Load training utilities
require('nn.StochasticGradient')

-- Utility functions
function nn.isTensor(obj)
   return torch.isTensor(obj)
end

function nn.type(obj)
   if type(obj) == 'table' and obj.__typename then
      return obj.__typename
   end
   return type(obj)
end

function nn.getParameters(module)
   -- Get all parameters and gradients as flat vectors
   local parameters = {}
   local gradParameters = {}
   
   module:parameters(parameters, gradParameters)
   
   -- Flatten
   local flatParameters = nn.flatten(parameters)
   local flatGradParameters = nn.flatten(gradParameters)
   
   return flatParameters, flatGradParameters
end

function nn.flatten(tensors)
   -- Flatten a table of tensors into a single vector
   if #tensors == 0 then
      return torch.Tensor()
   end
   
   local totalSize = 0
   for i, tensor in ipairs(tensors) do
      totalSize = totalSize + tensor:nElement()
   end
   
   local flat = torch.Tensor(totalSize)
   local offset = 1
   
   for i, tensor in ipairs(tensors) do
      local n = tensor:nElement()
      flat[{{offset, offset + n - 1}}]:copy(tensor:view(-1))
      offset = offset + n
   end
   
   return flat
end

return nn
