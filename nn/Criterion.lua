--[[
   nn.Criterion - Base class for loss functions
]]--

local Criterion = torch.class('nn.Criterion')

function Criterion:__init()
   self.gradInput = torch.Tensor()
   self.output = 0
end

function Criterion:updateOutput(input, target)
   error('updateOutput method not implemented')
end

function Criterion:forward(input, target)
   return self:updateOutput(input, target)
end

function Criterion:updateGradInput(input, target)
   error('updateGradInput method not implemented')
end

function Criterion:backward(input, target)
   return self:updateGradInput(input, target)
end

function Criterion:type(type)
   assert(type, 'Criterion:type(type) expects a string')
   
   -- Convert all tensors
   for key, param in pairs(self) do
      if torch.isTensor(param) then
         self[key] = param:type(type)
      end
   end
   
   return self
end

function Criterion:float()
   return self:type('torch.FloatTensor')
end

function Criterion:double()
   return self:type('torch.DoubleTensor')
end

function Criterion:cuda()
   return self:type('torch.CudaTensor')
end

return Criterion
