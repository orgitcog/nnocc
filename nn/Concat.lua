--[[
   nn.Concat - Container that concatenates outputs of child modules
]]--

local Concat, parent = torch.class('nn.Concat', 'nn.Container')

function Concat:__init(dimension)
   parent.__init(self)
   self.dimension = dimension or 1
   self.size = torch.LongStorage()
end

function Concat:updateOutput(input)
   local nModule = #self.modules
   
   -- Collect outputs from all modules
   local outputs = {}
   for i = 1, nModule do
      outputs[i] = self.modules[i]:updateOutput(input)
   end
   
   -- Concatenate along dimension
   self.output = torch.cat(outputs, self.dimension)
   
   return self.output
end

function Concat:updateGradInput(input, gradOutput)
   local nModule = #self.modules
   
   -- Split gradOutput for each module
   local offset = 1
   self.gradInput:resizeAs(input):zero()
   
   for i = 1, nModule do
      local outputSize = self.modules[i].output:size(self.dimension)
      local narrowGradOutput = gradOutput:narrow(self.dimension, offset, outputSize)
      local gradInput = self.modules[i]:updateGradInput(input, narrowGradOutput)
      self.gradInput:add(gradInput)
      offset = offset + outputSize
   end
   
   return self.gradInput
end

function Concat:accGradParameters(input, gradOutput, scale)
   local nModule = #self.modules
   local offset = 1
   
   for i = 1, nModule do
      local outputSize = self.modules[i].output:size(self.dimension)
      local narrowGradOutput = gradOutput:narrow(self.dimension, offset, outputSize)
      self.modules[i]:accGradParameters(input, narrowGradOutput, scale)
      offset = offset + outputSize
   end
end

return Concat
