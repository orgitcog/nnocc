--[[
   nn.Module - Base class for all neural network modules
   
   All modules should inherit from this class and implement:
   - updateOutput(input): compute output from input
   - updateGradInput(input, gradOutput): compute gradient w.r.t. input
   - accGradParameters(input, gradOutput, scale): accumulate parameter gradients
]]--

local Module = torch.class('nn.Module')

function Module:__init()
   self.gradInput = torch.Tensor()
   self.output = torch.Tensor()
   self._type = 'torch.DoubleTensor'
end

function Module:parameters()
   -- Return parameters and gradient parameters
   -- Override in subclasses that have parameters
   return {}, {}
end

function Module:updateOutput(input)
   -- Compute output from input
   -- Must be overridden in subclasses
   error('updateOutput method not implemented')
end

function Module:forward(input)
   -- Forward pass - compute output
   return self:updateOutput(input)
end

function Module:updateGradInput(input, gradOutput)
   -- Compute gradient w.r.t. input
   -- Must be overridden in subclasses  
   error('updateGradInput method not implemented')
end

function Module:backward(input, gradOutput, scale)
   -- Backward pass
   scale = scale or 1
   self:updateGradInput(input, gradOutput)
   self:accGradParameters(input, gradOutput, scale)
   return self.gradInput
end

function Module:accGradParameters(input, gradOutput, scale)
   -- Accumulate gradient w.r.t. parameters
   -- Override in subclasses that have parameters
end

function Module:accUpdateGradParameters(input, gradOutput, lr)
   -- Accumulate gradient and update parameters directly
   local gradWeight = self.gradWeight
   local gradBias = self.gradBias
   self.gradWeight = self.weight
   self.gradBias = self.bias
   self:accGradParameters(input, gradOutput, -lr)
   self.gradWeight = gradWeight
   self.gradBias = gradBias
end

function Module:sharedAccUpdateGradParameters(input, gradOutput, lr)
   -- For shared parameters (e.g., weight sharing)
   if self:parameters() then
      self:zeroGradParameters()
      self:accGradParameters(input, gradOutput, 1)
      self:updateParameters(lr)
   end
end

function Module:zeroGradParameters()
   -- Zero all parameter gradients
   local params, gradParams = self:parameters()
   if gradParams then
      for i, gradParam in ipairs(gradParams) do
         gradParam:zero()
      end
   end
end

function Module:updateParameters(learningRate)
   -- Update parameters using gradients
   local params, gradParams = self:parameters()
   if params then
      for i = 1, #params do
         params[i]:add(-learningRate, gradParams[i])
      end
   end
end

function Module:training()
   -- Set module to training mode
   self.train = true
end

function Module:evaluate()
   -- Set module to evaluation mode
   self.train = false
end

function Module:clone(...)
   -- Deep copy the module
   local f = torch.MemoryFile("rw"):binary()
   f:writeObject(self)
   f:seek(1)
   local clone = f:readObject()
   f:close()
   if select('#', ...) > 0 then
      clone:share(self, ...)
   end
   return clone
end

function Module:type(type, tensorCache)
   -- Convert module to a different tensor type
   assert(type, 'Module:type(type) expects a string')
   tensorCache = tensorCache or {}
   
   -- Convert all tensors in the module
   for key, param in pairs(self) do
      if torch.isTensor(param) then
         if not tensorCache[torch.pointer(param)] then
            tensorCache[torch.pointer(param)] = true
            self[key] = param:type(type)
         end
      elseif type(param) == 'table' then
         self[key] = self:_type_recursive(param, type, tensorCache)
      end
   end
   
   self._type = type
   return self
end

function Module:_type_recursive(tbl, type, tensorCache)
   -- Recursively convert tensors in tables
   local out = {}
   for k, v in pairs(tbl) do
      if torch.isTensor(v) then
         if not tensorCache[torch.pointer(v)] then
            tensorCache[torch.pointer(v)] = true
            out[k] = v:type(type)
         else
            out[k] = v
         end
      elseif type(v) == 'table' then
         out[k] = self:_type_recursive(v, type, tensorCache)
      else
         out[k] = v
      end
   end
   return out
end

function Module:float()
   return self:type('torch.FloatTensor')
end

function Module:double()
   return self:type('torch.DoubleTensor')
end

function Module:cuda()
   return self:type('torch.CudaTensor')
end

function Module:reset(stdv)
   -- Reset parameters
   -- Override in subclasses that have parameters
end

function Module:getParameters()
   -- Get all parameters as a flat vector
   -- and corresponding gradients
   
   local function storageInSet(set, storage)
      local storageAndOffset = set[torch.pointer(storage)]
      if storageAndOffset == nil then
         return nil
      end
      local _, offset = unpack(storageAndOffset)
      return offset
   end
   
   -- Collect unique parameters
   local params, gradParams = self:parameters()
   
   if not params or #params == 0 then
      return torch.Tensor(), torch.Tensor()
   end
   
   -- Calculate total size
   local totalSize = 0
   for i = 1, #params do
      totalSize = totalSize + params[i]:nElement()
   end
   
   -- Create flat tensors
   local flatParams = params[1].new(totalSize)
   local flatGradParams = gradParams[1].new(totalSize)
   
   -- Copy parameters
   local offset = 1
   for i = 1, #params do
      local nElement = params[i]:nElement()
      flatParams[{{offset, offset + nElement - 1}}]:copy(params[i]:view(-1))
      flatGradParams[{{offset, offset + nElement - 1}}]:copy(gradParams[i]:view(-1))
      offset = offset + nElement
   end
   
   return flatParams, flatGradParams
end

function Module:__tostring__()
   return torch.type(self)
end

return Module
