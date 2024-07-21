import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split

df = pd.read_csv("Diabetes Binary Classification.csv")
data = torch.from_numpy(df.values).float()

x = data[:,:-1]
y = data[:,-1:]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#model
class binary(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(binary, self).__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.linear = nn.Linear(input_dim, output_dim)
    self.act = nn.Sigmoid()
  
  def forward(self, x):
    y = self.act(self.linear(x))
    return y

model = binary(input_dim=x.shape[-1], output_dim=y.shape[-1])
optimizer = optim.Adam(model.parameters())

log_data = []

n_epochs = 1000
#print_interval = 50
for i in range(n_epochs):
  y_hat = model(x_train)
  loss = F.binary_cross_entropy(y_hat, y_train)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  '''
  if (i+1) % print_interval == 0:
    print('Epoch: %d, loss: %.4e' % (i+1, loss))
  '''

  #round to nearest ten thousandth  
  w = model.linear.weight.data.numpy()
  b = model.linear.bias.data.numpy()
  formatted_l = round(loss.item(), 4)
  formatted_w = [[round(float(value), 4) for value in row] for row in w]
  formatted_b = [round(float(value), 4) for value in b]
  
  format = {
    "epoch": i+1,
    "training_loss": formatted_l,
	  "weights": formatted_w,
   	"bias": formatted_b
  }
  log_data.append(format)
  
with open('file.json', 'w') as f:
  json.dump(log_data, f, indent=2)
  
  
#"training_loss": loss.item(),
#"weights": model.linear.weight.data.numpy().tolist(),
#"bias": model.linear.bias.data.numpy().tolist()
