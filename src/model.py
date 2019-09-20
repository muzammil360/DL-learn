from models.alexnet import alexnet

def getModel(_model_identifier):
	print("This is model function")

	if _model_identifier=='alexnet':
		model = alexnet()

	return model