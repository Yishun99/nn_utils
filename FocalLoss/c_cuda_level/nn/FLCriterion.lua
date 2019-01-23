local THNN = require 'nn.THNN'
local FLCriterion, parent = torch.class('nn.FLCriterion', 'nn.Criterion')

function FLCriterion:__init(alpha, gamma, sizeAverage)
   parent.__init(self)
   self.alpha = alpha
   self.gamma = gamma
   if sizeAverage ~= nil then
      self.sizeAverage = sizeAverage
   else
      self.sizeAverage = true
   end
end

function FLCriterion:updateOutput(input, target)
   -- pt = x * y + (1-x) * (1-y)
   -- alpha_factor = alpha * y + (1-alpha) * (1-y)
   -- w = alpha_factor * (1-pt)^gamma
   -- loss = -w (y*log(x) + (1-y)*log(1-x))
   assert( input:nElement() == target:nElement(),
   "input and target size mismatch")
   self.output_tensor = self.output_tensor or input.new(1)

   input.THNN.FLCriterion_updateOutput(
      input:cdata(),
      target:cdata(),
      self.output_tensor:cdata(),
      self.sizeAverage,
      self.alpha,
      self.gamma
   )

   self.output = self.output_tensor[1]
   return self.output
end

function FLCriterion:updateGradInput(input, target)
   -- bce_output = -(y*log(x) + (1-y)*log(1-x))
   -- alpha_factor = alpha * y + (1-alpha) * (1-y)
   -- gradient = alpha_factor * pow(1-pt,v-1) * (gamma*(1-2y)*bce_output + (1-pt)*((x-y)/((1-x)*x)))
   assert( input:nElement() == target:nElement(),
   "input and target size mismatch")

   input.THNN.FLCriterion_updateGradInput(
      input:cdata(),
      target:cdata(),
      self.gradInput:cdata(),
      self.sizeAverage,
      self.alpha,
      self.gamma
   )

   return self.gradInput
end
