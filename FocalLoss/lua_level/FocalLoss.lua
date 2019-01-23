local FocalLoss, parent = torch.class('FocalLoss', 'nn.Criterion')


function FocalLoss:__init()
	parent.__init(self)
	self.sizeAverage = true
	self.alpha = 0.25
	self.gamma = 2
	self.EPS = 1e-12
end


function FocalLoss:updateOutput(x, y)
	assert(x:nElement() == y:nElement(), "x and y size mismatch")

	self.pt = torch.cmul(x, y):add((1 - x):cmul(1 - y))
	self.alpha_factor = (1 - y):mul(1 - self.alpha):add(y * self.alpha)
	w = (1 - self.pt):pow(self.gamma):cmul(self.alpha_factor)

	self.bce_output = -(torch.log(x + self.EPS):cmul(y):add((1 + self.EPS - x):log():cmul(1 - y)))

	self.output = w:cmul(self.bce_output):mean()

	return self.output
end


function FocalLoss:updateGradInput(x, y)
	assert(x:nElement() == y:nElement(), "x and y size mismatch")
	if self.sizeAverage then
		norm = 1. / x:nElement()
	else
		norm = 1.
	end

	tmp = 1 - self.pt

	self.gradInput = self.alpha_factor
						 :cmul(torch.pow(tmp, self.gamma - 1))
						 :cmul((x - y):cdiv((1 + self.EPS - x):cmul(x + self.EPS)):cmul(tmp)
									  :add((1 - 2*y):cmul(self.bce_output):mul(self.gamma)))
						 :mul(norm)

	return self.gradInput
end

return FocalLoss
