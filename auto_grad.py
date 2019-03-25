import torch
import torch.nn.functional as F
import torch.utils.data as Data

BATCH_SIZE = 100

torch.manual_seed(7)

train_data = torch.load("train0325.pt").requires_grad_(True)
train_y = torch.load("energy0325.pt").requires_grad_(True)
input_size = train_data.size()[1]


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input = torch.nn.Linear(input_size, 2048)
        self.hidden_1 = torch.nn.Linear(2048, 1024)
        self.hidden_2 = torch.nn.Linear(1024, 512)
        self.hidden_3 = torch.nn.Linear(512, 256)
        self.output = torch.nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = F.relu(self.hidden_3(x))
        x = self.output(x)
        return x


# x_train = torch.rand((10000, 3), requires_grad=True)

net = Net()

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()

torch_data_set = Data.TensorDataset(train_data, train_y)
loader = Data.DataLoader(
    dataset=torch_data_set,
    batch_size=64,
    shuffle=True
)

for epoch in range(50):
    for step, (batch_x, energy) in enumerate(loader):

        optimizer.zero_grad()
        prediction_temp = net(batch_x.float())
        loss_1 = loss_func(prediction_temp, energy.float())
        # loss_1.backward(retain_graph=True)

        # prediction = torch.autograd.grad(prediction_temp.sum(), batch_x, create_graph=True)
        # loss_2 = 20.0 * loss_func(-prediction[0], batch_y)
        print('Epoch: ', epoch, '| Step: ', step, '| loss: ', loss_1.data.numpy())
        loss_1.backward()
        optimizer.step()

print(train_y.float(), net(train_data.float()))

