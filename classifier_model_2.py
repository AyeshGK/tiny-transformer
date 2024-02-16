import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import predictive_coding as pc

torch.manual_seed(1234)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocab_size = 20000

block_size = 200
embeds_size = 100
num_classes = 2
drop_prob = 0.13
batch_size = 32
epochs = 30
num_heads = 4
head_size = embeds_size // num_heads
model_path = '/content/checkpoint/model_classification.pth'
model_loader = False

class DataSet(Dataset):
	def __init__(self, mode):
		super().__init__()
		self.imdb_data = torch.tensor(torch.load(f"/content/tiny-transformer/imdb/imdb_{mode}.json"))

	def __getitem__(self, idx):
		data = self.imdb_data[idx]
		# The last element is the target
		seq = data[:-1]
		targets = torch.zeros(num_classes, device=device)
		targets[data[-1]] = 1
		return seq, targets

	def __len__(self):
		return len(self.imdb_data)


dataset_X = DataSet('train')
dataset_y = DataSet('test')

train_data = DataLoader(dataset_X, batch_size, shuffle=True)
test_data = DataLoader(dataset_y, shuffle=False)


class block(nn.Module):
	def __init__(self):
		super(block, self).__init__()
		self.attention = nn.MultiheadAttention(embeds_size, num_heads, batch_first=True)
		self.ffn = nn.Sequential(
			nn.Linear(embeds_size, 2 * embeds_size),
			nn.LeakyReLU(),
			nn.Linear(2 * embeds_size, embeds_size),
		)
		self.drop1 = nn.Dropout(drop_prob)
		self.drop2 = nn.Dropout(drop_prob)
		self.ln1 = nn.LayerNorm(embeds_size)
		self.ln2 = nn.LayerNorm(embeds_size)

	def forward(self, hidden_state):
		attn, _ = self.attention(hidden_state, hidden_state, hidden_state, need_weights=False)
		attn = self.drop1(attn)
		out = self.ln1(hidden_state + attn)
		observed = self.ffn(out)
		observed = self.drop2(observed)
		return self.ln2(out + observed)


class transformer(nn.Module):
	def __init__(self):
		super(transformer, self).__init__()

		self.tok_emb = nn.Embedding(vocab_size, embeds_size)
		self.pos_emb = nn.Embedding(block_size, embeds_size)
		self.block = block()
		self.pcl = pc.PCLayer()
		self.ln1 = nn.LayerNorm(embeds_size)
		self.ln2 = nn.LayerNorm(embeds_size)

		self.classifier_head = nn.Sequential(
			nn.Linear(embeds_size, embeds_size),
			nn.LeakyReLU(),
			nn.Dropout(drop_prob),
			nn.Linear(embeds_size, embeds_size),
			nn.LeakyReLU(),
			nn.Linear(embeds_size, num_classes),
			nn.Softmax(dim=1),
		)

		print("number of parameters: %.2fM" % (self.num_params()/1e6,))

	def num_params(self):
		n_params = sum(p.numel() for p in self.parameters())
		return n_params

	def forward(self, seq):
		# print("seq :",seq)
		B,T = seq.shape
		embedded = self.tok_emb(seq)
		embedded = embedded + self.pos_emb(torch.arange(T, device=device))
		output = self.block(embedded)
		output = output.mean(dim=1)
		output = self.pcl(output)
		output = self.classifier_head(output)
		return output



# PCTrainer_kwargs:
# 	update_x_at: "all"
# 	optimizer_x_fn: "SGD"
# 	optimizer_x_kwargs:
# 		lr: 0.5
# 	x_lr_discount: 0.5
# 	x_lr_amplifier: 1.0

# 	update_p_at: "all"
# 	optimizer_p_fn: "Adam"
# 	optimizer_p_kwargs:
# 		lr:
# 			grid_search:
# 				# - 0.005
# 				# - 0.001
# 				# - 0.0005
# 				- 0.00025
# 				- 0.0001
# 				- 0.000075
# 				- 0.00005
# 				- 0.000025
# 				- 0.00001
# 				# - 0.000005
# 				# - 0.000001
# 				# - 0.0000005
# 		weight_decay:
# 			grid_search:
# 				# - 0.0
# 				# - 0.001
# 				- 0.01
# 				# - 0.1
# 				# - 1.0
# train_on_batch_kwargs:
# 	is_log_progress: False
# 	is_return_results_every_t: False
# 	# debug
# 	# is_return_results_every_t: True
# 	is_checking_after_callback_after_t: False
	

config={
	'PCTrainer_kwargs': {
		'update_x_at': "all",
		'optimizer_x_fn': "SGD",
		'optimizer_x_kwargs': {
			'lr': 0.5,
		},
		'x_lr_discount': 0.5,
		'x_lr_amplifier': 1.0,
		'update_p_at': "all",
		'optimizer_p_fn': "Adam",
		'optimizer_p_kwargs': {
			'lr': 0.00025,
			'weight_decay': 0.01,
		},
	},
	'train_on_batch_kwargs': {
		'is_log_progress': False,
		'is_return_results_every_t': False,
		'is_checking_after_callback_after_t': False,
	},
}

class UModel:
	def __init__(self, config):
		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		# self.model = u.create_model(
		# 	self.config['predictive_coding'],
		# 	**self.config['model'],
		# ).to(self.device)
		self.model = transformer().to(self.device)


		self.config['PCTrainer_kwargs']['optimizer_x_fn']=eval(
			'torch.optim.{}'.format(self.config['PCTrainer_kwargs']['optimizer_x_fn'])
		)
		self.config['PCTrainer_kwargs']['optimizer_p_fn']=eval(
			'torch.optim.{}'.format(self.config['PCTrainer_kwargs']['optimizer_p_fn'])
		)
		# self.config['PCTrainer_kwargs']['plot_progress_at']=eval(
		# 	self.config['PCTrainer_kwargs']['plot_progress_at']
		# )
		self.pc_trainer = pc.PCTrainer(
			self.model,
			**self.config['PCTrainer_kwargs'],
		)

		self.pc_trainer = pc.PCTrainer(
            self.model,
            **self.config['PCTrainer_kwargs'],
        )

		self.model_loss = nn.BCEWithLogitsLoss()
		self.model_optimizer = torch.optim.RMSprop(self.model.parameters(), lr=4e-4)

	def loss_fn(self,outputs, target):
		print("loss fn called")
		return (outputs - target).pow(2).sum() * 0.5

	def train(self, train_data, test_data):
		for epoch in range(epochs):
			losses = 0
			for (inputs, targets) in train_data:
				inputs = inputs.to(device)
				targets = targets.to(device)
				output = self.model(inputs)
				loss = self.model_loss(output, targets)
				self.model_optimizer.zero_grad()
				loss.backward()
				self.model_optimizer.step()
				losses += loss.item()
			print(f'[{epoch}][Train]', ', losses', losses)
			self.model.eval()
			test_loss = 0
			passed = 0
			for (inputs, targets) in test_data:
				with torch.no_grad():
					inputs = inputs.to(device)
					targets = targets.to(device)
					outputs = self.model(inputs)
					if outputs.argmax() == targets.argmax():
						passed += 1
			self.model.train()
			print("train data:",train_data)
			self.pc_trainer.train_on_batch(
				# data, loss_fn,
				train_data,self.loss_fn,
				loss_fn_kwargs={
					'target': targets,
				},
            	**self.config['train_on_batch_kwargs'],
        	)
			print("=================:",train_data)
			print(f'[{epoch}][Test]', ', accuracy', passed / len(dataset_y))

		torch.save(self.model.state_dict(), model_path)

# model = transformer()
# if model_loader:
# 	model.load_state_dict(torch.load(model_path))
# model.to(device)
		
# model = 


# model_loss = nn.BCEWithLogitsLoss()
# model_optimizer = torch.optim.RMSprop(model.parameters(), lr=4e-4)

# for epoch in range(epochs):
# 	losses = 0
# 	for (inputs, targets) in train_data:
# 		inputs = inputs.to(device)
# 		targets = targets.to(device)
# 		output = model(inputs)
# 		loss = model_loss(output, targets)
# 		model_optimizer.zero_grad()
# 		loss.backward()
# 		model_optimizer.step()
# 		losses += loss.item()
# 	print(f'[{epoch}][Train]', ', losses', losses)
# 	model.eval()
# 	test_loss = 0
# 	passed = 0
# 	for (inputs, targets) in test_data:
# 		with torch.no_grad():
# 			inputs = inputs.to(device)
# 			targets = targets.to(device)
# 			outputs = model(inputs)
# 			if outputs.argmax() == targets.argmax():
# 				passed += 1
# 	model.train()
# 	print(f'[{epoch}][Test]', ', accuracy', passed / len(dataset_y))
		

		
umodel = UModel(config)
umodel.train(train_data, test_data)
torch.save(umodel.model.state_dict(), model_path)
