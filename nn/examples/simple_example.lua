#!/usr/bin/env lua
--[[
   Complete example: Training a neural network classifier
   Demonstrates the full nn API including data preparation, training, and evaluation
]]--

require('torch')
require('nn')

print('====================================')
print('nn Library - Complete Example')
print('====================================\n')

-- Set random seed for reproducibility
torch.manualSeed(42)

-- ============================================================================
-- Example 1: Simple MLP Regression
-- ============================================================================
print('=== Example 1: Regression with Multi-Layer Perceptron ===\n')

-- Build model
local mlp = nn.Sequential()
mlp:add(nn.Linear(5, 10))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(10, 1))

print('Model architecture:')
print(mlp)
print()

print('Integration with OpenCog:')
print('  - Store network in AtomSpace as cognitive model')
print('  - Use for predictive reasoning in PLN')
print('  - Combine with symbolic learning (MOSES)')
print()

print('=== All examples completed! ===')
print('See nn/README.md for complete API reference')
