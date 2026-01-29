--[[
   nn.Linear - Fully connected linear layer
   
   y = Wx + b
   
   where:
   - W is weight matrix (outputSize x inputSize)
   - b is bias vector (outputSize)
]]--

local Linear, parent = torch.class('nn.Linear', 'nn.Module')

function Linear:__init(inputSize, outputSize, bias)
   parent.__init(self)
   
   self.weight = torch.Tensor(outputSize, inputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   
   if bias ~= false then
      self.bias = torch.Tensor(outputSize)
      self.gradBias = torch.Tensor(outputSize)
   end
   
   self:reset()
end

function Linear:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1.0 / math.sqrt(self.weight:size(2))
   end
   
   self.weight:uniform(-stdv, stdv)
   if self.bias then
      self.bias:uniform(-stdv, stdv)
   end
end

function Linear:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.weight:size(1))
      if self.bias then
         self.output:copy(self.bias)
      else
         self.output:zero()
      end
      self.output:addmv(1, self.weight, input)
   elseif input:dim() == 2 then
      local nFrame = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nFrame, self.weight:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      if self.bias then
         self.output:addr(1, input.new(nFrame):fill(1), self.bias)
      end
      self.output:addmm(1, input, self.weight:t())
   else
      error('input must be vector or matrix')
   end
   
   return self.output
end

function Linear:updateGradInput(input, gradOutput)
   if self.gradInput then
      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      
      if input:dim() == 1 then
         self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
      elseif input:dim() == 2 then
         self.gradInput:addmm(0, 1, gradOutput, self.weight)
      end
      
      return self.gradInput
   end
end

function Linear:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   
   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
      if self.bias then
         self.gradBias:add(scale, gradOutput)
      end
   elseif input:dim() == 2 then
      self.gradWeight:addmm(scale, gradOutput:t(), input)
      if self.bias then
         self.gradBias:addmv(scale, gradOutput:t(), input.new(input:size(1)):fill(1))
      end
   end
end

function Linear:parameters()
   if self.bias then
      return {self.weight, self.bias}, {self.gradWeight, self.gradBias}
   else
      return {self.weight}, {self.gradWeight}
   end
end

function Linear:__tostring__()
   return string.format('%s(%d -> %d)', torch.type(self), self.weight:size(2), self.weight:size(1))
end

return Linear
