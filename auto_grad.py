import torch
import torch.nn.functional as F
import torch.utils.data as Data

torch.manual_seed(7)
train_data = torch.rand((10000, 3), requires_grad=True)
add_noise = torch.randn((10000, 1))/5000.0
print(train_data)
# with torch.no_grad():
train_y = torch.reshape(torch.sum(train_data**2, dim=1)**0.5, (-1, 1)) + add_noise

v = torch.reshape(torch.ones(10000), (-1, 1))
train_y.backward(v)

y = train_data.grad
with torch.no_grad():
    train_y = train_y.clone()
print(train_y)
print(train_y.mean())


class Net(torch.nn.Module):
    def __init__(self, n_feature):
        super(Net, self).__init__()
        self.input = torch.nn.Linear(n_feature, 512)
        self.hidden_1 = torch.nn.Linear(512, 256)
        self.hidden_2 = torch.nn.Linear(256, 128)
        self.output = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = self.output(x)
        return x


x_train = torch.rand((10000, 3), requires_grad=True)

net = Net(n_feature=3)

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss(reduction="mean")

torch_dataset = Data.TensorDataset(x_train, train_y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=100,
    shuffle=True
)


v = torch.reshape(torch.ones(100), (-1, 1))

for t in range(50):
    for step, (batch_x, batch_y) in enumerate(loader):  # for each training step.
        prediction = net(batch_x)

        loss = loss_func(prediction, batch_y)
        print('Epoch: ', t, '| Step: ', step, '| loss: ', loss.data.numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

x_train_2 = torch.rand((10000, 3))
y_train_2 = net.forward(x_train_2)
print(y_train_2)

print(((y_train_2 - train_y)**2.0).sum()/10000.0)
