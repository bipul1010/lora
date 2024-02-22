## Importing Libraries
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
import sys


## Load MNIST Dataset

# Make torch deterministic
_ = torch.manual_seed(0)


# Load the MNIST dataset
class SimpleNeuralNet(nn.Module):
    def __init__(
        self, hidden_size_1: int = 500, hidden_size_2=200, output_classes=10
    ) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(28 * 28, hidden_size_1)
        self.linear_2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear_3 = nn.Linear(hidden_size_2, output_classes)
        self.relu = nn.ReLU()

    def forward(self, img):
        ## (B,1,28,28) -> (B,28*28)
        x = img.view(-1, 28 * 28)
        x = self.relu(self.linear_1(x))  ## (B,28*28) -> (B,200)
        x = self.relu(self.linear_2(x))  ## (B,200) -> (B,200)
        x = self.linear_3(x)  ##(B,200) -> (B,10)
        return x


def get_ds():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    mnist_trainset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    # print(mnist_trainset[30][0].shape)
    train_loader = DataLoader(mnist_trainset, shuffle=True, batch_size=10)

    # for train_batch in train_loader:
    #     X,Y = train_batch[0],train_batch[1]
    #     print(X,Y)
    #     break
    # classes = [int(y) for x in train_loader for y in x[1]]
    output_classes = 10  ## len(set(classes))

    mnist_testset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(mnist_testset, shuffle=True, batch_size=10)
    return (train_loader, test_loader, output_classes)


def train(train_loader, model, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-9)
    global_step = 0

    for epoch in range(0, epochs):
        torch.cuda.empty_cache()

        model.train()

        batch_iterator = tqdm(train_loader, desc=f"Processing Epoch: {epoch}")
        losses = []

        for train_batch in batch_iterator:
            X, Y = train_batch[0], train_batch[1]  ##X -> (B,1,28,28) ; Y -> (B)
            logits = model(X)  ## (B,10)

            loss = F.cross_entropy(logits, Y)
            batch_iterator.set_postfix({"loss": loss.item()})

            losses.append(loss.item())

            optimizer.zero_grad(set_to_none=True)

            ##back propagate the loss and compute the gradient
            loss.backward()

            ## update the weight
            optimizer.step()

            global_step += 1

        batch_iterator.write(
            f"Epoch :{epoch} | Avg. Training Loss: {sum(losses) / len(losses)}"
        )
    return model


def test(test_loader, model):
    model.eval()
    total = 0
    correct_match = 0
    incorrect_match = {i: 0 for i in range(0, 10)}
    for test_batch in test_loader:
        X, Y = test_batch[0], test_batch[1]

        logits = model(X)  ## (B,10)
        value, predicted_labels = torch.max(logits, dim=1)
        for y_predicted, y in zip(predicted_labels, Y):
            if y_predicted == y:
                correct_match += 1
            else:
                y = int(y)
                incorrect_match[y] += 1
            total += 1

    acc = correct_match / total
    print(incorrect_match)
    print(f"Accu: {acc}")
    for digit in sorted(list(incorrect_match.keys())):
        print(f"No of Incorrect Match of Digit {digit}: {incorrect_match[digit]}")
    return acc


def load_model(model_filename):
    model = SimpleNeuralNet(hidden_size_1=500, hidden_size_2=200, output_classes=10)
    model.load_state_dict(torch.load(model_filename))
    return model


train_loader, test_loader, output_classes = get_ds()
if __name__ == "__main__":
    train_loader, test_loader, output_classes = get_ds()
    model_path = Path("./model_path")
    model_filename = str(model_path / "simple_net.pt")
    if sys.argv[1] == "train":
        model = SimpleNeuralNet(
            hidden_size_1=500, hidden_size_2=200, output_classes=output_classes
        )
        model = train(train_loader=train_loader, model=model)
        torch.save(model.state_dict(), model_filename)
    if sys.argv[1] == "test":
        load_model(model_filename)
        test(test_loader=test_loader, model=model)
