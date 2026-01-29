# nn Integration with OpenCog Collection

This document describes how the `nn` neural network library integrates with the OpenCog Collection (OCC) cognitive architecture.

## Overview

The `nn` library provides neural network capabilities that complement OpenCog's symbolic AI components, enabling hybrid neuro-symbolic reasoning and cognitive synergy.

## Architecture Integration

### 1. AtomSpace Integration

Neural network models can be represented and stored in the AtomSpace hypergraph:

```scheme
; Store network architecture
(DefineLink
  (ConceptNode "MLPClassifier")
  (SequentialNode
    (LinearNode (List (Number 784) (Number 128)))
    (ActivationNode "ReLU")
    (LinearNode (List (Number 128) (Number 10)))
    (ActivationNode "LogSoftMax")))

; Store learned parameters
(StateLink
  (ConceptNode "MLPClassifier")
  (FloatValueNode [weight_vector]))
```

### 2. Cognitive Agent Framework (a9nn)

The `nn` library serves as the foundation for the **a9nn** cognitive agent framework described in `.github/agents/a9nn.md`:

- **nn.Module**: Base for all cognitive modules
- **nn.CognitiveAgent**: Multi-agent orchestration
- **nn.NeuroAgent**: Personality-driven behavior
- **nn.NNECCOAgent**: Full NNECCO architecture

### 3. Hybrid Reasoning (PLN + nn)

Combine Probabilistic Logic Networks (PLN) with neural networks:

```lua
-- Neural network provides pattern recognition
local classifier = nn.Sequential()
classifier:add(nn.Linear(features, hidden))
classifier:add(nn.ReLU())
classifier:add(nn.Linear(hidden, classes))

-- PLN provides logical inference
-- DeductionRule: (A→B) ∧ (B→C) ⇒ (A→C)
-- Network outputs feed into PLN inference chains
```

### 4. Attention-Guided Learning

Integration with OpenCog's attention allocation mechanism:

```lua
-- Get high-attention atoms from ECAN
local importantAtoms = getHighSTIAtoms()

-- Convert to features for neural network
local features = atomsToFeatures(importantAtoms)

-- Train network on attention-weighted data
local output = network:forward(features)
```

### 5. Evolutionary Learning (MOSES)

MOSES can optimize neural network architectures:

```lua
-- Define network architecture as MOSES program
-- MOSES evolves:
--   - Number of layers
--   - Layer sizes
--   - Activation functions
--   - Connection patterns

-- Fitness function: network accuracy + complexity penalty
```

## Integration Points

### CogGML Microkernel

Neural networks integrate with CogGML's cognitive loop:

1. **Perception**: nn processes sensory input
2. **Reasoning**: Output feeds into PLN inference
3. **Learning**: MOSES optimizes network structure
4. **Action**: Network output drives behavior

### CogSelf AGI Framework

The nn library supports CogSelf's cognitive synergy goals:

- **Pattern Recognition**: Neural networks identify patterns
- **Symbolic Reasoning**: PLN performs logical inference  
- **Hybrid Learning**: Combine neural and symbolic learning
- **Emergent Intelligence**: Synergy from component interaction

### Distributed AtomSpace

Neural networks can run across distributed AtomSpace shards:

```lua
-- Shard 1: Perceptual processing
local perceptualNet = nn.Sequential()
perceptualNet:add(nn.Linear(raw_input_size, features))

-- Shard 2: Reasoning integration
local reasoningNet = nn.Sequential()
reasoningNet:add(nn.Linear(features, symbolic_space))

-- Communication via AtomSpace-DAS
```

## Use Cases

### 1. Cognitive Classification

Neural network classifies sensory input, PLN reasons about classifications:

```lua
-- Vision: Classify objects
local vision = nn.Sequential()
vision:add(nn.Linear(image_pixels, 512))
vision:add(nn.ReLU())
vision:add(nn.Linear(512, object_classes))
vision:add(nn.LogSoftMax())

-- Result stored in AtomSpace for reasoning
-- (ClassificationLink (ImageNode "img1") (ConceptNode "cat"))
```

### 2. Reinforcement Learning Agents

Combine nn policy networks with symbolic goal reasoning:

```lua
-- Policy network
local policy = nn.Sequential()
policy:add(nn.Linear(state_dim, 128))
policy:add(nn.Tanh())
policy:add(nn.Linear(128, action_dim))
policy:add(nn.SoftMax())

-- Symbolic goal: (GoalNode "maximize_reward")
-- PLN reasons about subgoals
-- Policy network selects actions
```

### 3. Natural Language Processing

Neural language models + symbolic knowledge:

```lua
-- Word embeddings
local embedder = nn.Sequential()
embedder:add(nn.LookupTable(vocab_size, embedding_dim))

-- Sequence processing
local encoder = nn.Sequential()
encoder:add(embedder)
encoder:add(nn.LSTM(embedding_dim, hidden_dim))

-- Output feeds into RelEx for syntactic/semantic parsing
-- Then into PLN for logical reasoning about text
```

### 4. Predictive Models

Learn temporal patterns for cognitive prediction:

```lua
-- Learn sequences
local predictor = nn.Sequential()
predictor:add(nn.Linear(history_size, 64))
predictor:add(nn.Tanh())
predictor:add(nn.Linear(64, future_size))

-- Predictions inform attention allocation
-- High-uncertainty predictions get more attention
```

## API Integration Examples

### AtomSpace Bridge

```lua
-- Load network from AtomSpace
function loadNetworkFromAtomSpace(atomspace, networkName)
   local architecture = atomspace:getNode(networkName)
   local network = buildFromArchitecture(architecture)
   local weights = atomspace:getValue(networkName, "weights")
   network:setParameters(weights)
   return network
end

-- Save network to AtomSpace  
function saveNetworkToAtomSpace(atomspace, network, networkName)
   local architecture = describeArchitecture(network)
   atomspace:addNode(networkName, architecture)
   local weights = network:getParameters()
   atomspace:setValue(networkName, "weights", weights)
end
```

### CogServer Interface

```lua
-- Register network as CogServer module
cogserver.registerModule("Classifier", {
   initialize = function()
      network = nn.Sequential()
      -- Build network...
   end,
   
   execute = function(atomspace, input)
      local features = extractFeatures(atomspace, input)
      local output = network:forward(features)
      storeResults(atomspace, output)
   end
})
```

### Attention Integration

```lua
-- Attention-weighted learning
function trainWithAttention(network, atomspace, dataset)
   for i = 1, #dataset do
      local input = dataset[i].input
      local target = dataset[i].target
      
      -- Get attention value for this datum
      local sti = atomspace:getSTI(dataset[i].atom)
      local weight = sigmoid(sti / 100.0)
      
      -- Weighted gradient update
      local output = network:forward(input)
      network:backward(input, gradOutput)
      network:updateParameters(learningRate * weight)
   end
end
```

## Performance Considerations

### Memory Sharing

Neural networks share memory with AtomSpace efficiently:

- **Tensor Storage**: Store as FloatValues in AtomSpace
- **Zero-Copy**: Access tensors without duplication
- **Batch Processing**: Process multiple atoms in parallel

### Distributed Computing

Networks can be distributed across cognitive shards:

- **Model Parallelism**: Split layers across shards
- **Data Parallelism**: Replicate model, split data
- **Pipeline Parallelism**: Pipeline layer execution

### GPU Acceleration

When available, use CUDA tensors for acceleration:

```lua
-- Convert to GPU
network:cuda()
input = input:cuda()

-- Forward/backward on GPU
output = network:forward(input)
```

## Future Enhancements

### Planned Integrations

1. **Convolutional Networks**: For vision processing
2. **Recurrent Networks**: For temporal reasoning
3. **Attention Mechanisms**: Neural attention + ECAN attention
4. **Graph Neural Networks**: Process AtomSpace graphs directly
5. **Meta-Learning**: Learn to learn via MOSES + nn

### Research Directions

- **Neuro-Symbolic Unification**: Seamless neural-symbolic reasoning
- **Explainable Neural Networks**: Symbolic explanations of neural decisions
- **Cognitive Architectures**: Full AGI integration
- **Self-Modifying Networks**: Networks that evolve their own structure

## References

- [nn README](../README.md): Complete API reference
- [a9nn Agent](.github/agents/a9nn.md): Cognitive agent framework
- [OpenCog PLN](../pln/): Probabilistic Logic Networks
- [MOSES](../asmoses/): Evolutionary learning
- [AtomSpace](../atomspace/): Hypergraph knowledge representation

## Contributing

Contributions to nn-OCC integration are welcome! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

Areas for contribution:
- New layer types
- AtomSpace integration utilities
- Cognitive agent modules
- Performance optimizations
- Example applications
