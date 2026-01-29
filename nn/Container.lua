--[[
   nn.Container - Base class for modules containing other modules
]]--

local Container, parent = torch.class('nn.Container', 'nn.Module')

function Container:__init(...)
   parent.__init(self)
   self.modules = {}
end

function Container:add(module)
   table.insert(self.modules, module)
   return self
end

function Container:get(index)
   return self.modules[index]
end

function Container:size()
   return #self.modules
end

function Container:parameters()
   local params = {}
   local gradParams = {}
   
   for i = 1, #self.modules do
      local p, gp = self.modules[i]:parameters()
      if p then
         for j = 1, #p do
            table.insert(params, p[j])
            table.insert(gradParams, gp[j])
         end
      end
   end
   
   return params, gradParams
end

function Container:training()
   parent.training(self)
   for i = 1, #self.modules do
      self.modules[i]:training()
   end
end

function Container:evaluate()
   parent.evaluate(self)
   for i = 1, #self.modules do
      self.modules[i]:evaluate()
   end
end

function Container:type(type, tensorCache)
   parent.type(self, type, tensorCache)
   for i = 1, #self.modules do
      self.modules[i]:type(type, tensorCache)
   end
   return self
end

function Container:zeroGradParameters()
   for i = 1, #self.modules do
      self.modules[i]:zeroGradParameters()
   end
end

function Container:updateParameters(learningRate)
   for i = 1, #self.modules do
      self.modules[i]:updateParameters(learningRate)
   end
end

function Container:reset(stdv)
   for i = 1, #self.modules do
      self.modules[i]:reset(stdv)
   end
end

function Container:__tostring__()
   local str = torch.type(self) .. ' {\n'
   for i = 1, #self.modules do
      str = str .. '  (' .. i .. '): ' .. tostring(self.modules[i]):gsub('\n', '\n  ') .. '\n'
   end
   str = str .. '}'
   return str
end

return Container
