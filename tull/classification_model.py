
class ClassificationModel:
	def forward(self, embedding_positive, embedding_negative):
		"""
		trains the model one iteration
		"""
		assert self.weights != None		
		assert self.anchor_embedding != None		

		raise NotImplementedError()

	def get_weights(self):
		raise NotImplementedError()
		# return a vector of model weights

	def set_weights(self):
		raise NotImplementedError()

	def set_anchor_embedding(self):
		raise NotImplementedError()
		
	def match(self, embedding):
		"""
		returns a fraction of how much two embeddings matches
		- 1.0 is 100%
		- 0.0 is   0%
		- 0.5 is  50%
		"""
		raise NotImplementedError()
		return 0.5
		
