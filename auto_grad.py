import torch
import torch.nn.functional as F

torch.manual_seed(7)
train_data = torch.rand((10000, 3), requires_grad=True)
add_noise = torch.randn((10000, 1))/500.0
print(train_data)
# with torch.no_grad():
train_y = torch.reshape(torch.sum(train_data**2, dim=1)**0.5, (-1, 1))

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
        self.input = torch.nn.Linear(n_feature, 64)
        self.hidden = torch.nn.Linear(64, 32)
        self.output = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x


x_train = torch.rand((10000, 3), requires_grad=True)

net = Net(n_feature=3)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss(reduction="sum")

for t in range(50):
    for j in range(0, 10000, 10):
        train_temp = x_train[j: j + 10]
        prediction = net(train_temp)

        loss = loss_func(prediction, train_y[j: j + 10])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(prediction, train_y[j: j + 10])
        if j == 5000:
            print(loss.data.numpy())

print(net(x_train))
