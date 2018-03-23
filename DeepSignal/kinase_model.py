import torch
import torch.nn as nn

class EncoderKinase(nn.Module):
	def __init__(self, input_size, hidden_size, 
				input_dropout_p=0, dropout_p=0, n_layers=1, 
				bidirectional=True, rnn_cell='lstm'):
		super(EncoderKinase, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.input_dropout_p = input_dropout_p
		self.dropout_p = dropout_p
		self.n_layers = n_layers
		self.rnn_cell = rnn_cell.lower()
		if self.rnn_cell == 'lstm':
			self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers,
							batch_first=True, bidirectional=bidirectional,
							dropout=dropout_p)
		elif self.rnn_cell == 'gru':
			self.rnn = nn.GRU(hidden_size, hidden_size, n_layers,
							batch_first=True, bidirectional=bidirectional,
							dropout=dropout_p)
		else:
			raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))
		self.embedding = nn.Embedding(input_size, hidden_size)
		self.input_dropout = nn.Dropout(p=input_dropout_p)


	def forward(self, input_var):
		embedded = self.embedding(input_var)
		embedded = self.input_dropout(embedded)
		output, hidden = self.rnn(embedded)
		if self.rnn_cell == 'lstm':
			hidden = hidden[0]
		return output, hidden

class MLP(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(MLP, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, output_size)
		self.log_softmax = nn.functional.log_softmax

	def forward(self, input_var):
		output = self.fc1(input_var)
		output = self.relu(output)
		output = self.fc2(output)
		output = self.log_softmax(output)
		return output

class ListModule(object):
	#Should work with all kind of module
	def __init__(self, module, prefix, *args):
		self.module = module
		self.prefix = prefix
		self.num_module = 0
		for new_module in args:
			self.append(new_module)

	def append(self, new_module):
		if not isinstance(new_module, nn.Module):
			raise ValueError('Not a Module')
		else:
			self.module.add_module(self.prefix + str(self.num_module), new_module)
			self.num_module += 1

	def __len__(self):
		return self.num_module

	def __getitem__(self, i):
		if i < 0 or i >= self.num_module:
			raise IndexError('Out of bound')
		return getattr(self.module, self.prefix + str(i))

class DecoderKinase(nn.Module):
	r"""
	Outputs: decoder_outputs, None, ret_dict
		- decoder_outputs: (out_seq_len, batch_size, output_size)
		- ret_dict
			- sequences: 
	"""
	def __init__(self, output_size, hidden_size, out_seq_len=15, mlp_hidden_size=64):
		super(DecoderKinase, self).__init__()
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.out_seq_len = out_seq_len
		# self.out = [MLP(hidden_size, mlp_hidden_size, output_size)
		# 				for _ in range(out_seq_len)]
		# self.transfer = MLP(hidden_size, mlp_hidden_size, output_size)
		self.out = ListModule(self, 'out_')
		for _ in range(out_seq_len):
			self.out.append(MLP(hidden_size, mlp_hidden_size, output_size))

	def forward(self, encoder_hidden):
		decoder_outputs = []
		sequence_symbols = []
		ret_dict = dict()
		batch_size = encoder_hidden.size(1)
		for out in self.out:
			# print('size', self.hidden_size, batch_size, encoder_hidden.size())
			step_out = out(encoder_hidden.view(-1, self.hidden_size)).view(batch_size, self.output_size)
			decoder_outputs.append(step_out)	# (batch_size, output_size)
			symbols = decoder_outputs[-1].topk(1)[1]	# (batch_size)
			sequence_symbols.append(symbols)
		# print('hidden', encoder_hidden.size())
		ret_dict['sequence'] = sequence_symbols
		ret_dict['length'] = [self.out_seq_len] * batch_size
		return decoder_outputs, None, ret_dict

class KinaseModel(nn.Module):
	def __init__(self, encoder, decoder):
		super(KinaseModel, self).__init__()
		self.encoder = encoder
		self.decoder = decoder

	def flatten_parameters(self):
		self.encoder.rnn.flatten_parameters()
		# self.decoder.rnn.flatten_parameters()
		
	def forward(self, input_var, input_lengths=None, target_var=None):
		_, encoder_hidden = self.encoder(input_var)     
		# decoder_input = encoder_hidden.squeeze()
		# result = self.decoder(decoder_input)
		result = self.decoder(encoder_hidden)
		return result
