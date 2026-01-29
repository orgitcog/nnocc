--[[
   nn.MSECriterion - Mean Squared Error loss
   
   loss = (1/n) * sum((input - target)^2)
]]--

local MSECriterion, parent = torch.class('nn.MSECriterion', 'nn.Criterion')

function MSECriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function MSECriterion:updateOutput(input, target)
   self.output = input:clone():add(-1, target)
   self.output:cmul(self.output)
   
   if self.sizeAverage then
      self.output = self.output:sum() / input:nElement()
   else
      self.output = self.output:sum()
   end
   
   return self.output
end

function MSECriterion:updateGradInput(input, target)
   local norm = self.sizeAverage and 2.0 / input:nElement() or 2.0
   
   self.gradInput:resizeAs(input)
   self.gradInput:copy(input):add(-1, target)
   self.gradInput:mul(norm)
   
   return self.gradInput
end

return MSECriterion
