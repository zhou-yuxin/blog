# Pytorch实现MNIST手写数字识别

书接上回《[Pytorch初体验](../Pytorch初体验/README.md)》，今天实践一下用Pytorch实现一个神经网络，来实现手写数字识别。

首先实现一个神经网络。为了方便起见，我使用了简单粗暴的全连接神经网络，代码如下：
```python
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
```

MNIST数据集中的图片都是28x28的灰度图，所以网络的输入层就是28x28=784个节点。输出层共10个节点，每个节点对应0～9这10个数字的得分，得分最高的数字作为网络的最终输出。构造函数里的`hidden_layers`是一个整数列表，每一个整数表示从输入层与输出层之间的每一个隐藏层的节点个数。比如`MyMNIST([256, 32])`就是定义了一个784->256->32->10的四层全连接网络。可以看到，这个参数同时控制了网络的宽度和深度。如果`hidden_layers`为None，则没有隐藏层。由于隐藏层的个数是不定的，所以代码中必须使用循环动态创建每个隐藏层，并置于list中，方便后续使用。但是从《[Pytorch初体验](../Pytorch初体验/README.md)》中我们得知，处于容器类型内的`torch.nn.Module`实例的参数是不会被`parameters()`方法自动捕获的。为此，我使用了`setattr()`方法，将每个隐藏层都设置为当前对象的一个成员变量，名为`_hidden_layer_<i>`。这样，隐藏层既能被自动捕获，又能在list中享受操作的便利。

从`forward()`函数实现可以看出，隐藏层都是使用GELU激活函数。而输出层则不使用激活函数，因为分类问题只需要取得分最大的节点即可，加了激活函数不改变最大值所对应的节点。

接下来就要实例化网络、损失函数和优化器：
```python
hidden_layers = [256, 32]
learn_rate = 0.001
device = "cuda"

device = torch.device(device)
model = MyMNIST(hidden_layers).to(device = device)
loss_fn = torch.nn.CrossEntropyLoss().to(device = device)
optim = torch.optim.Adam(model.parameters(), lr = learn_rate)
```

代码中的`device`控制模型所在位置，可以选择"cpu"或者"cuda:0"这样的参数。

然后准备训练数据集、实现训练过程：
```python
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
```

训练数据集`dataset`可以像数组一样用下标索引，每一项是一个输入Tensor与对应标签的元组，比如：
```python
(x, gt) = dataset[0]
print(x)
print(gt)
```

会输出：
```
tensor([[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000, 0.0000, 0.0000],
          ...
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0706, 0.6706,
          0.8588, 0.9922, 0.9922, 0.9922, 0.9922, 0.7647, 0.3137, 0.0353,
          0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000, 0.0000, 0.0000],
          ...
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000, 0.0000, 0.0000]]])
5
```

训练过程中，我们通常是每次从数据集中选取一个小子集构成一个batch，一次性输入模型，输出一个batch的结果，从而加速训练（这是矩阵计算的特性，凑成batch做计算所需要的时间远小于一个一个做）。这个任务可以交给`torch.utils.data.DataLoader`，也就是代码中的`data_loader`实例。迭代`data_loader`，每次得到`batch_size`个输入与对应标签，直到所有数据都迭代完。它的构造函数中`shuffle = True`指定了每一轮迭代前，数据集都会被洗牌，也就是随机排列，这样每一轮所产生的batch都是不同，从而避免过拟合。

函数`train_step()`中，在`data_loader`上迭代每一个batch（`x`包含`batch_size`个图片Tensor，`gt`包含`batch_size`个标签），输入模型，得到`y`（`y`包含`batch_size`个Tensor，每个Tensor是10维，即输出层10个节点的值），然后计算损失值`loss`。再次强调，这个过程中，每一步是如何运算的都被Pytorch自动记录着，这样才能反向求导。在调用`loss.backward()`之后，这个过程中的所有参与运算的参数都拥有了一个对于`loss`标量的导数，即可用于更新参数。但是需要注意的是，**在Pytorch中，每次backward()之后，新的导数值都是累加在上次结果之上的。** 这样设计的好处是，一个batch中的所有训练数据的导数能够自动累加，得到一个“总体更新方向”，代码上以batch为单位训练与单个数据训练无异。但是在batch之间我们需要把上次batch得到的导数清空，这也就是`loss.backward()`之前需要调用`optim.zero_grad()`的原因。在得到导数后，调用`optim.step()`更新每一个参数。如果传入了`stat = True`，那么`train_step()`会统计一下这一轮训练中的平均准确度。

现在可以开始训练了：
```python
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
```

共`epoch = 50`轮训练。训练过程中，每5轮统计一下当前的精度。

最后可以保存模型，或者把模型导出为ONNX格式：
```python
torch.save(model, "mnist.model")

dummy_input = torch.randn([1, MyMNIST.IMG_SIZE]).to(device = device)
torch.onnx.export(model, dummy_input, "mnist.onnx",
        input_names = ["images"], output_names = ["numbers"],
        dynamic_axes = {"images": {0: "batch_size"}, "numbers": {0: "batch_size"}},
        export_params = True)
```

导出ONNX格式的`torch.onnx.export()`的第二个参数`dummy_input`稍难理解。导出ONNX的原理是这样的：向模型输入一个数据，执行一遍推理，这个过程中的所有计算过程会被onnx模块记录，作为ONNX流程图。这个`dummy_input`就是用于执行一遍推理的随机输入，没人关心最后是什么结果，只要Tensor的维度、尺寸正确即可。`input_names`和`output_names`是给这个模型的输入输出层取个名字，这样加载ONNX模型后，可以用名字来设置输入、读取结果。之所以设计为数组形式，是考虑到模型的`forward()`可能要接收多个Tensor，返回多个Tensor，所以名字需要与实际模型的输入输出一一对应。`dynamic_axes`指定了哪些Tensor的哪些维度是动态的，是一个dict，key是Tensor名字，value是另一个dict，其key是第几维度，value是这个维度的名称，比如上面代码中，输入的images的第0维是batch_size，输出的images的第0维是batch_size。

至此，我们就有了Pytorch的模型mnist.model，后续可以使用`torch.load()`重新加载。也有了`mnist.onnx`，作为一个通用格式，可以被其他框架（比如微软的onnxruntime、Intel的openvino和NVIDIA的TensorRT）加载。

完整代码在[train.py](./train.py)中。