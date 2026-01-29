--[[
   nn.SoftMax - SoftMax activation function
   
   f(x_i) = exp(x_i) / sum_j exp(x_j)
]]--

local SoftMax, parent = torch.class('nn.SoftMax', 'nn.Module')

function SoftMax:updateOutput(input)
   self.output:resizeAs(input)
   
   if input:dim() == 1 then
      local max_val = input:max()
      self.output:copy(input):add(-max_val):exp()
      self.output:div(self.output:sum())
   elseif input:dim() == 2 then
      local nFrame = input:size(1)
      local nDim = input:size(2)
      
      for i = 1, nFrame do
         local max_val = input[i]:max()
         self.output[i]:copy(input[i]):add(-max_val):exp()
         self.output[i]:div(self.output[i]:sum())
      end
   else
      error('input must be vector or matrix')
   end
   
   return self.output
end

function SoftMax:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   
   if input:dim() == 1 then
      local s = (self.output:dot(gradOutput))
      self.gradInput:copy(gradOutput):add(-s, self.output)
      self.gradInput:cmul(self.output)
   elseif input:dim() == 2 then
      local nFrame = input:size(1)
      for i = 1, nFrame do
         local s = self.output[i]:dot(gradOutput[i])
         self.gradInput[i]:copy(gradOutput[i]):add(-s, self.output[i])
         self.gradInput[i]:cmul(self.output[i])
      end
   end
   
   return self.gradInput
end

return SoftMax
