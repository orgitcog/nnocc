--[[
   nn.ClassNLLCriterion - Negative Log Likelihood loss
   
   Used with LogSoftMax for classification
   
   loss = -input[target]
]]--

local ClassNLLCriterion, parent = torch.class('nn.ClassNLLCriterion', 'nn.Criterion')

function ClassNLLCriterion:__init(weights)
   parent.__init(self)
   self.weights = weights
   self.sizeAverage = true
end

function ClassNLLCriterion:updateOutput(input, target)
   if input:dim() == 1 then
      -- Single sample
      self.output = -input[target]
      if self.weights then
         self.output = self.output * self.weights[target]
      end
   elseif input:dim() == 2 then
      -- Batch of samples
      local nFrame = input:size(1)
      self.output = 0
      for i = 1, nFrame do
         local t = target[i]
         local val = -input[i][t]
         if self.weights then
            val = val * self.weights[t]
         end
         self.output = self.output + val
      end
      if self.sizeAverage then
         self.output = self.output / nFrame
      end
   else
      error('input must be vector or matrix')
   end
   
   return self.output
end

function ClassNLLCriterion:updateGradInput(input, target)
   self.gradInput:resizeAs(input):zero()
   
   if input:dim() == 1 then
      -- Single sample
      self.gradInput[target] = -1
      if self.weights then
         self.gradInput[target] = self.gradInput[target] * self.weights[target]
      end
   elseif input:dim() == 2 then
      -- Batch of samples
      local nFrame = input:size(1)
      for i = 1, nFrame do
         local t = target[i]
         self.gradInput[i][t] = -1
         if self.weights then
            self.gradInput[i][t] = self.gradInput[i][t] * self.weights[t]
         end
      end
      if self.sizeAverage then
         self.gradInput:div(nFrame)
      end
   end
   
   return self.gradInput
end

return ClassNLLCriterion
