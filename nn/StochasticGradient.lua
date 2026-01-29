--[[
   nn.StochasticGradient - Simple trainer using stochastic gradient descent
]]--

local StochasticGradient = torch.class('nn.StochasticGradient')

function StochasticGradient:__init(module, criterion)
   self.module = module
   self.criterion = criterion
   self.learningRate = 0.01
   self.learningRateDecay = 0
   self.maxIteration = 25
   self.shuffleIndices = true
   self.verbose = true
end

function StochasticGradient:train(dataset)
   local iteration = 1
   local currentLearningRate = self.learningRate
   local module = self.module
   local criterion = self.criterion
   
   local shuffledIndices = torch.randperm(#dataset, 'torch.LongTensor')
   if not self.shuffleIndices then
      for i = 1, #dataset do
         shuffledIndices[i] = i
      end
   end
   
   if self.verbose then
      print('# StochasticGradient: training')
   end
   
   while true do
      local currentError = 0
      for t = 1, #dataset do
         local idx = shuffledIndices[t]
         local sample = dataset[idx]
         local input = sample[1]
         local target = sample[2]
         
         -- Forward pass
         local output = module:forward(input)
         local err = criterion:forward(output, target)
         currentError = currentError + err
         
         -- Backward pass
         module:zeroGradParameters()
         local df_do = criterion:backward(output, target)
         module:backward(input, df_do)
         
         -- Update parameters
         module:updateParameters(currentLearningRate)
      end
      
      currentError = currentError / #dataset
      
      if self.verbose then
         print(string.format('# iter %d current error = %f', iteration, currentError))
      end
      
      iteration = iteration + 1
      currentLearningRate = self.learningRate / (1 + iteration * self.learningRateDecay)
      
      if iteration > self.maxIteration then
         break
      end
   end
end

return StochasticGradient
