--[[
   nn.LogSoftMax - Log(SoftMax) activation
   
   f(x_i) = log(exp(x_i) / sum_j exp(x_j))
          = x_i - log(sum_j exp(x_j))
]]--

local LogSoftMax, parent = torch.class('nn.LogSoftMax', 'nn.Module')

function LogSoftMax:updateOutput(input)
   self.output:resizeAs(input)
   
   if input:dim() == 1 then
      local max_val = input:max()
      self.output:copy(input):add(-max_val)
      local logsum = math.log(torch.exp(self.output):sum())
      self.output:add(-logsum)
   elseif input:dim() == 2 then
      local nFrame = input:size(1)
      for i = 1, nFrame do
         local max_val = input[i]:max()
         self.output[i]:copy(input[i]):add(-max_val)
         local logsum = math.log(torch.exp(self.output[i]):sum())
         self.output[i]:add(-logsum)
      end
   else
      error('input must be vector or matrix')
   end
   
   return self.output
end

function LogSoftMax:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   
   if input:dim() == 1 then
      local s = gradOutput:sum()
      self.gradInput:copy(gradOutput):add(-s, torch.exp(self.output))
   elseif input:dim() == 2 then
      local nFrame = input:size(1)
      for i = 1, nFrame do
         local s = gradOutput[i]:sum()
         self.gradInput[i]:copy(gradOutput[i]):add(-s, torch.exp(self.output[i]))
      end
   end
   
   return self.gradInput
end

return LogSoftMax
