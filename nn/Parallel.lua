--[[
   nn.Parallel - Container that applies each child module to
   the corresponding input slice and concatenates their outputs
]]--

local Parallel, parent = torch.class('nn.Parallel', 'nn.Container')

function Parallel:__init(inputDimension, outputDimension)
   parent.__init(self)
   self.inputDimension = inputDimension
   self.outputDimension = outputDimension
   self.size = torch.LongStorage()
end

function Parallel:updateOutput(input)
   local nModule = #self.modules
   
   -- Split input along inputDimension
   local slices = input:split(1, self.inputDimension)
   
   -- Process each slice through corresponding module
   local outputs = {}
   for i = 1, nModule do
      outputs[i] = self.modules[i]:updateOutput(slices[i]:squeeze(self.inputDimension))
   end
   
   -- Concatenate outputs along outputDimension
   self.output = torch.cat(outputs, self.outputDimension)
   
   return self.output
end

function Parallel:updateGradInput(input, gradOutput)
   local nModule = #self.modules
   local slices = input:split(1, self.inputDimension)
   local gradSlices = gradOutput:split(1, self.outputDimension)
   
   self.gradInput:resizeAs(input)
   local gradInputSlices = self.gradInput:split(1, self.inputDimension)
   
   for i = 1, nModule do
      local gradInput = self.modules[i]:updateGradInput(
         slices[i]:squeeze(self.inputDimension),
         gradSlices[i]:squeeze(self.outputDimension)
      )
      gradInputSlices[i]:copy(gradInput)
   end
   
   return self.gradInput
end

function Parallel:accGradParameters(input, gradOutput, scale)
   local nModule = #self.modules
   local slices = input:split(1, self.inputDimension)
   local gradSlices = gradOutput:split(1, self.outputDimension)
   
   for i = 1, nModule do
      self.modules[i]:accGradParameters(
         slices[i]:squeeze(self.inputDimension),
         gradSlices[i]:squeeze(self.outputDimension),
         scale
      )
   end
end

return Parallel
