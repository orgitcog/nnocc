--[[
   nn.Sigmoid - Sigmoid activation function
   
   f(x) = 1 / (1 + exp(-x))
]]--

local Sigmoid, parent = torch.class('nn.Sigmoid', 'nn.Module')

function Sigmoid:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   self.output:sigmoid()
   return self.output
end

function Sigmoid:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   -- d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
   self.gradInput:fill(1):add(-1, self.output):cmul(self.output)
   self.gradInput:cmul(gradOutput)
   return self.gradInput
end

return Sigmoid
