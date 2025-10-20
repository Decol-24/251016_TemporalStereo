import torch

import re

def FunctionSoftsplat(tenInput, tenFlow, tenMetric, strType):
	assert(tenMetric is None or tenMetric.shape[1] == 1)
	assert(strType in ['summation', 'average', 'linear', 'softmax'])

	if strType == 'average':
		tenInput = torch.cat([ tenInput, tenInput.new_ones(tenInput.shape[0], 1, tenInput.shape[2], tenInput.shape[3]) ], 1)

	elif strType == 'linear':
		tenInput = torch.cat([ tenInput * tenMetric, tenMetric ], 1)

	elif strType == 'softmax':
		tenInput = torch.cat([ tenInput * tenMetric.exp(), tenMetric.exp() ], 1)

	# end

	tenOutput = _FunctionSoftsplat.apply(tenInput, tenFlow)

	if strType != 'summation':
		tenNormalize = tenOutput[:, -1:, :, :]

		# tenNormalize[tenNormalize == 0.0] = 1.0

		tenOutput = tenOutput[:, :-1, :, :] / (tenNormalize + 1e-22)
	# end

	return tenOutput
# end