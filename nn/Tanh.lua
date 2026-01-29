--[[
   nn.Tanh - Hyperbolic tangent activation function
   
   f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
]]--

local Tanh, parent = torch.class('nn.Tanh', 'nn.Module')

function Tanh:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   self.output:tanh()
   return self.output
end

function Tanh:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   -- d/dx tanh(x) = 1 - tanh(x)^2
   self.gradInput:fill(1):add(-1, torch.pow(self.output, 2))
   self.gradInput:cmul(gradOutput)
   return self.gradInput
end

return Tanh
