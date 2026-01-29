--[[
   nn.CrossEntropyCriterion - Cross Entropy loss
   
   Combines LogSoftMax and ClassNLLCriterion
   
   loss = -log(exp(input[target]) / sum(exp(input)))
]]--

local CrossEntropyCriterion, parent = torch.class('nn.CrossEntropyCriterion', 'nn.Criterion')

function CrossEntropyCriterion:__init(weights)
   parent.__init(self)
   self.weights = weights
   self.sizeAverage = true
   self.logSoftMax = nn.LogSoftMax()
   self.nll = nn.ClassNLLCriterion(weights)
   self.nll.sizeAverage = self.sizeAverage
end

function CrossEntropyCriterion:updateOutput(input, target)
   -- Apply LogSoftMax
   local logProbs = self.logSoftMax:forward(input)
   
   -- Apply NLL
   self.output = self.nll:forward(logProbs, target)
   
   return self.output
end

function CrossEntropyCriterion:updateGradInput(input, target)
   -- Backprop through NLL
   local gradLogProbs = self.nll:backward(self.logSoftMax.output, target)
   
   -- Backprop through LogSoftMax
   self.gradInput = self.logSoftMax:backward(input, gradLogProbs)
   
   return self.gradInput
end

return CrossEntropyCriterion
