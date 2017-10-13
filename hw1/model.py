class Model:
	def __init__(self, training_matrix, init_weight,  init_bias, init_learning_rate, regular_factor):
		self.training_matrix = training_matrix
		self.weight = init_weight
		self.bias = init_bias
		self.learning_rate = init_learning_rate
		self.regular_factor = regular_factor
		self.error = 0
	def reset(self, init_weight, init_bias,init_learning_rate, regular_factor):
		self.weight = init_weight
		self.bias = init_bias
		self.learning_rate = learning_rate
		self.regular_factor = regular_factor
		self.error = 0