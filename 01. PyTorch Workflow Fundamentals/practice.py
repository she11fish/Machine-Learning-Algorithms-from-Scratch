import torch
from torch import nn
import matplotlib.pyplot as plt

weight = 0.3
bias = 0.9
X = torch.arange(0, 100, 0.02).unsqueeze(dim=1)
y = weight * X + bias
train_split = int(len(X) * 0.8)
x_train, x_test, y_train, y_test = X[:train_split], X[train_split:], y[:train_split], y[train_split:]
torch.manual_seed(42)
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
torch.manual_seed(42)

model = LinearRegression()
loss_fn = nn.L1Loss()
optim = torch.optim.SGD(params=model.parameters(), lr=0.001)
torch.manual_seed(42)

epochs = 1000
for epoch in range(epochs):
    model.train()
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    optim.zero_grad()
    loss.backward()
    optim.step()
    model.eval()
    with torch.inference_mode():
        test_pred = model(x_test)
        test_loss = loss_fn(test_pred, y_test)
    if epoch % 20 == 0:
        print(f"Loss: {loss}, Test Loss: {test_loss}")
        # print(model.state_dict())
model.eval()
print(f"{model.state_dict()}")
# plt.scatter(X.numpy(), y.numpy(), label='Original data')
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(X.numpy(), y.numpy(), X.numpy(), (model.linear_layer.weight * X + model.linear_layer.bias).detach().numpy(), label='True Regression Line', linestyle='--') 
plt.show()
print([2, 3, 3][:, 0])