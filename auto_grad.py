import torch
import torch.nn.functional as F
import torch.utils.data as Data

BATCH_SIZE = 100

torch.manual_seed(7)
train_data = torch.rand((10000, 3), requires_grad=True)
print(train_data.max(), train_data.min())
add_noise = torch.randn((10000, 1))/5000.0
print(train_data)
train_y = torch.reshape(torch.sum(train_data**2.0, dim=1), (-1, 1))

v = torch.reshape(torch.ones(10000), (-1, 1))
train_y.backward(v)

y = -train_data.grad
with torch.no_grad():
    y = y.clone()
    train_y = train_y.clone()
print(y)


class Net(torch.nn.Module):
    def __init__(self, n_feature):
        super(Net, self).__init__()
        self.input = torch.nn.Linear(n_feature, 32)
        self.hidden_1 = torch.nn.Linear(32, 16)
        self.output = torch.nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden_1(x))
        x = self.output(x)
        return x


# x_train = torch.rand((10000, 3), requires_grad=True)

net = Net(n_feature=3)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()

torch_data_set = Data.TensorDataset(train_data, train_y, y)
loader = Data.DataLoader(
    dataset=torch_data_set,
    batch_size=100,
    shuffle=True
)


for epoch in range(50):
    for step, (batch_x, energy, batch_y) in enumerate(loader):

        optimizer.zero_grad()
        prediction_temp = net(batch_x)
        loss_1 = loss_func(prediction_temp, energy)
        loss_1.backward(retain_graph=True)

        prediction = torch.autograd.grad(prediction_temp.sum(), batch_x, create_graph=True)
        loss_2 = 20.0 * loss_func(-prediction[0], batch_y)
        print('Epoch: ', epoch, '| Step: ', step, '| loss: ', loss_2.data.numpy())
        (loss_1 + loss_2).backward()
        optimizer.step()


y_train_2_temp = net.forward(train_data)
y_train_2 = net.forward(train_data).sum()
force = -torch.autograd.grad(y_train_2, train_data)[0]
print(force)
print(y)

print(((y_train_2_temp - train_y)**2.0).sum()/10000.0)
print(((y - force)**2.0).sum()/10000.0)

