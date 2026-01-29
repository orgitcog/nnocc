#!/usr/bin/env lua
--[[
   Basic tests for nn modules
]]--

require('torch')
require('nn')

print('Testing nn library...\n')

-- Test Linear module
print('=== Testing nn.Linear ===')
local linear = nn.Linear(10, 5)
local input = torch.randn(10)
local output = linear:forward(input)
print('Input size:', input:size(1))
print('Output size:', output:size(1))
assert(output:size(1) == 5, 'Linear output size mismatch')
print('✓ Linear forward pass OK\n')

-- Test activation functions
print('=== Testing Activation Functions ===')

local tanh = nn.Tanh()
local tanhOut = tanh:forward(torch.Tensor{-1, 0, 1})
print('Tanh(-1, 0, 1):', tanhOut)
assert(math.abs(tanhOut[2]) < 0.01, 'Tanh(0) should be ~0')
print('✓ Tanh OK')

local sigmoid = nn.Sigmoid()
local sigOut = sigmoid:forward(torch.Tensor{0})
print('Sigmoid(0):', sigOut[1])
assert(math.abs(sigOut[1] - 0.5) < 0.01, 'Sigmoid(0) should be ~0.5')
print('✓ Sigmoid OK')

local relu = nn.ReLU()
local reluOut = relu:forward(torch.Tensor{-1, 0, 1})
print('ReLU(-1, 0, 1):', reluOut)
assert(reluOut[1] == 0, 'ReLU(-1) should be 0')
assert(reluOut[3] == 1, 'ReLU(1) should be 1')
print('✓ ReLU OK\n')

-- Test Sequential container
print('=== Testing nn.Sequential ===')
local mlp = nn.Sequential()
mlp:add(nn.Linear(10, 5))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(5, 1))

local input = torch.randn(10)
local output = mlp:forward(input)
print('MLP input size:', input:size(1))
print('MLP output size:', output:size(1))
assert(output:size(1) == 1, 'MLP output size should be 1')
print('✓ Sequential OK\n')

-- Test backward pass
print('=== Testing Backward Pass ===')
local gradOutput = torch.ones(1)
local gradInput = mlp:backward(input, gradOutput)
print('GradInput size:', gradInput:size(1))
assert(gradInput:size(1) == 10, 'GradInput size mismatch')
print('✓ Backward pass OK\n')

-- Test MSE Criterion
print('=== Testing nn.MSECriterion ===')
local criterion = nn.MSECriterion()
local pred = torch.Tensor{1, 2, 3}
local target = torch.Tensor{1, 2, 3}
local loss = criterion:forward(pred, target)
print('MSE loss (identical):', loss)
assert(loss < 0.01, 'MSE should be ~0 for identical tensors')

local pred2 = torch.Tensor{1, 2, 3}
local target2 = torch.Tensor{2, 3, 4}
local loss2 = criterion:forward(pred2, target2)
print('MSE loss (off by 1):', loss2)
assert(loss2 > 0, 'MSE should be positive for different tensors')
print('✓ MSECriterion OK\n')

-- Test parameter management
print('=== Testing Parameter Management ===')
local params, gradParams = mlp:parameters()
print('Number of parameter tensors:', #params)
assert(#params > 0, 'Should have parameters')
assert(#params == #gradParams, 'Should have matching gradient parameters')

mlp:zeroGradParameters()
for i = 1, #gradParams do
   assert(gradParams[i]:sum() == 0, 'Gradients should be zero')
end
print('✓ Parameter management OK\n')

-- Test training mode
print('=== Testing Training Mode ===')
mlp:training()
print('Training mode set')
mlp:evaluate()
print('Evaluation mode set')
print('✓ Mode switching OK\n')

print('=== All tests passed! ===')
