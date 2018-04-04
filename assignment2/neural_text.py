# Code in file autograd/two_layer_net_custom_function.py
import torch
from torch.autograd import Variable


class MyReLU(torch.autograd.Function):
	"""
	We can implement our own custom autograd Functions by subclassing
	torch.autograd.Function and implementing the forward and backward passes
	which operate on Tensors.
	"""

	def forward(self, input):
		"""
		In the forward pass we receive a Tensor containing the input and return a
		Tensor containing the output. You can cache arbitrary Tensors for use in the
		backward pass using the save_for_backward method.
		"""
		self.save_for_backward(input)
		return input.clamp(min=0)

	def backward(self, grad_output):
		"""
		In the backward pass we receive a Tensor containing the gradient of the loss
		with respect to the output, and we need to compute the gradient of the loss
		with respect to the input.
		"""
		input, = self.saved_tensors
		grad_input = grad_output.clone()
		grad_input[input < 0] = 0
		return grad_input


dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs, and wrap them in Variables.
x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

# Create random Tensors for weights, and wrap them in Variables.
w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

learning_rate = 1e-6
for t in range(500):
	# Construct an instance of our MyReLU class to use in our network
	relu = MyReLU()

	# Forward pass: compute predicted y using operations on Variables; we compute
	# ReLU using our custom autograd operation.
	y_pred = relu(x.mm(w1)).mm(w2)
	print('accuracy: {0}'.format((y == y_pred).sum()/N))
	# Compute and print loss
	loss = (y_pred - y).pow(2).sum()
	print(t, loss.data[0])

	# Use autograd to compute the backward pass.
	loss.backward()

	# Update weights using gradient descent
	w1.data -= learning_rate * w1.grad.data
	w2.data -= learning_rate * w2.grad.data

	# Manually zero the gradients after running the backward pass
	w1.grad.data.zero_()
	w2.grad.data.zero_()

	# Code in file nn/two_layer_net_module.py


import torch
from torch.autograd import Variable


class TwoLayerNet(torch.nn.Module):
	def __init__(self, D_in, H, D_out):
		"""
		In the constructor we instantiate two nn.Linear modules and assign them as
		member variables.
		"""
		super(TwoLayerNet, self).__init__()
		self.linear1 = torch.nn.Linear(D_in, H)
		self.linear2 = torch.nn.Linear(H, D_out)

	def forward(self, x):
		"""
		In the forward function we accept a Variable of input data and we must return
		a Variable of output data. We can use Modules defined in the constructor as
		well as arbitrary operators on Variables.
		"""
		h_relu = self.linear1(x).clamp(min=0)
		y_pred = self.linear2(h_relu)
		return y_pred
