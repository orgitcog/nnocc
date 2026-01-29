--[[
   nn.Sequential - Container that chains modules sequentially
   
   Output of one module becomes input to the next
]]--

local Sequential, parent = torch.class('nn.Sequential', 'nn.Container')

function Sequential:__init()
   parent.__init(self)
end

function Sequential:updateOutput(input)
   local currentOutput = input
   for i = 1, #self.modules do
      currentOutput = self.modules[i]:updateOutput(currentOutput)
   end
   self.output = currentOutput
   return self.output
end

function Sequential:updateGradInput(input, gradOutput)
   local currentGradOutput = gradOutput
   for i = #self.modules, 1, -1 do
      local currentModule = self.modules[i]
      local previousModule = self.modules[i - 1]
      local currentInput
      if i == 1 then
         currentInput = input
      else
         currentInput = previousModule.output
      end
      currentGradOutput = currentModule:updateGradInput(currentInput, currentGradOutput)
   end
   self.gradInput = currentGradOutput
   return self.gradInput
end

function Sequential:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   
   local currentGradOutput = gradOutput
   for i = #self.modules, 1, -1 do
      local currentModule = self.modules[i]
      local previousModule = self.modules[i - 1]
      local currentInput
      if i == 1 then
         currentInput = input
      else
         currentInput = previousModule.output
      end
      currentModule:accGradParameters(currentInput, currentGradOutput, scale)
      currentGradOutput = currentModule.gradInput
   end
end

function Sequential:backward(input, gradOutput, scale)
   scale = scale or 1
   local currentGradOutput = gradOutput
   for i = #self.modules, 1, -1 do
      local currentModule = self.modules[i]
      local previousModule = self.modules[i - 1]
      local currentInput
      if i == 1 then
         currentInput = input
      else
         currentInput = previousModule.output
      end
      currentGradOutput = currentModule:backward(currentInput, currentGradOutput, scale)
   end
   self.gradInput = currentGradOutput
   return self.gradInput
end

return Sequential
