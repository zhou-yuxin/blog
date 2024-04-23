import time
import torch
import torchvision
import torch.utils.data

class MyMNIST(torch.nn.Module):

    IMG_SIZE = 28 * 28

    def __init__(self, hidden_layers: tuple = None):
        super().__init__()
        self._hidden_layers = []
        input_size = self.IMG_SIZE
        if hidden_layers is not None:
            for (i, node_num) in enumerate(hidden_layers):
                layer = torch.nn.Linear(input_size, node_num)
                self._hidden_layers.append(layer)
                # make the layer able to be auto captured by parameters()
                setattr(self, "_hidden_layer_" + str(i), layer)
                input_size = node_num
        self._output_layer = torch.nn.Linear(input_size, 10)

    def forward(self, x: torch.Tensor):
        # the first axis is dynamic batch_size
        x = x.view(-1, self.IMG_SIZE)
        for layer in self._hidden_layers:
            x = torch.nn.functional.gelu(layer(x))
        return self._output_layer(x)

hidden_layers = [256, 32]
learn_rate = 0.001
device = "cuda"

device = torch.device(device)
model = MyMNIST(hidden_layers).to(device = device)
loss_fn = torch.nn.CrossEntropyLoss().to(device = device)
optim = torch.optim.Adam(model.parameters(), lr = learn_rate)

batch_size = 128

dataset = torchvision.datasets.MNIST(root = "./dataset", train = True, download = True,
        transform = torchvision.transforms.ToTensor())
data_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)

def train_step(stat: bool = True):
    correct = 0
    total = 0
    for (x, gt) in data_loader:
        x = x.to(device = device)
        gt = gt.to(device = device)
        y = model(x)
        loss: torch.Tensor = loss_fn(y, gt)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if stat:
            with torch.no_grad():
                y = torch.argmax(y, dim = 1)
                equal: torch.Tensor = y == gt
                correct += equal.sum().item()
                total += y.size(0)
    return correct / total if stat else 0

epoch = 50

start_time = time.time_ns()
for i in range(1, epoch + 1):
    # stat the quality every 5 epoches
    stat = (i % 5) == 0
    accuracy = train_step(stat)
    if stat:
        print("iteration = %d, accuracy = %.5f" % (i, accuracy))
end_time = time.time_ns()
print("time = %.4f s" % ((end_time - start_time) / 1000000000))

torch.save(model, "mnist.model")

dummy_input = torch.randn([1, MyMNIST.IMG_SIZE]).to(device = device)
torch.onnx.export(model, dummy_input, "mnist.onnx",
        input_names = ["images"], output_names = ["numbers"],
        dynamic_axes = {"images": {0: "batch_size"}, "numbers": {0: "batch_size"}},
        export_params = True)
