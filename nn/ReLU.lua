--[[
   nn.ReLU - Rectified Linear Unit activation
   
   f(x) = max(0, x)
]]--

local ReLU, parent = torch.class('nn.ReLU', 'nn.Module')

function ReLU:__init(inplace)
   parent.__init(self)
   self.inplace = inplace or false
   self.threshold = 0
   self.val = 0
end

function ReLU:updateOutput(input)
   if self.inplace then
      self.output:set(input)
   else
      self.output:resizeAs(input):copy(input)
   end
   self.output:threshold(self.threshold, self.val)
   return self.output
end

function ReLU:updateGradInput(input, gradOutput)
   if self.inplace then
      self.gradInput:set(gradOutput)
   else
      self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   end
   -- Gradient is 1 if output > 0, else 0
   self.gradInput:cmul(torch.gt(self.output, self.threshold):type(self.output:type()))
   return self.gradInput
end

return ReLU
