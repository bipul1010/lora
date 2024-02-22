import torch
import sys
import torch.nn as nn
from pathlib import Path
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from mnist_model import SimpleNeuralNet, load_model, train, test_loader, test


model_path = Path("./model_path")
model_filename = str(model_path / "simple_net.pt")
lora_filename = str(model_path / "lora_states.pt")


def get_ds_for_finetuning():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    mnist_trainset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    exclude_indices = (mnist_trainset.targets == 9) | (mnist_trainset.targets == 8)

    mnist_trainset.data = mnist_trainset.data[exclude_indices]
    mnist_trainset.targets = mnist_trainset.targets[exclude_indices]
    train_loader = DataLoader(mnist_trainset, batch_size=10, shuffle=True)
    return train_loader


class LoraLayer(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, rank: int, alpha: int
    ) -> None:
        super().__init__()
        ##x = (B,in_dim); W = (out_dim,in_dim); y = xW^T + b- >(B,out_dim)
        ## x @ B @ A-> (B,in_dim) @ (in_dim,r) @ (r,out_dim)
        ## B = gaussian normalization, A = zeros

        self.lora_A = nn.Parameter(torch.zeros(rank, out_features))
        self.lora_B = nn.Parameter(torch.randn(in_features, rank))

        self.scale = alpha / rank

    def forward(self, x):
        return torch.matmul(x, torch.matmul(self.lora_B, self.lora_A)) * self.scale


class LinearwithLora(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: int) -> None:
        super().__init__()

        self.linear = linear
        self.lora = LoraLayer(
            in_features=self.linear.in_features,
            out_features=self.linear.out_features,
            rank=rank,
            alpha=alpha,
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def adapt_lora_layers(model: SimpleNeuralNet, layers: list, rank: int, alpha: int):
    for x in layers:
        if x == "linear_1":
            model.linear_1 = LinearwithLora(model.linear_1, rank=rank, alpha=alpha)
        elif x == "linear_2":
            model.linear_2 = LinearwithLora(model.linear_2, rank=rank, alpha=alpha)
        elif x == "linear_3":
            model.linear_3 = LinearwithLora(model.linear_3, rank=rank, alpha=alpha)
    return model


def save_lora_weights(model: SimpleNeuralNet, layers: list, lora_filename: str):
    save_weights = dict()
    for x in layers:
        if x == "linear_1":
            save_weights[x] = model.linear_1.lora.state_dict()
        elif x == "linear_2":
            save_weights[x] = model.linear_2.lora.state_dict()
        elif x == "linear_3":
            save_weights[x] = model.linear_3.lora.state_dict()
    torch.save(save_weights, lora_filename)


def load_lora_weights(model: SimpleNeuralNet, layers: list, lora_filename: str):
    lora_state_dict = torch.load(lora_filename)
    for x in layers:
        if x == "linear_1":
            model.linear_1.lora.load_state_dict(lora_state_dict["linear_1"])
        elif x == "linear_2":
            model.linear_2.lora.load_state_dict(lora_state_dict["linear_2"])
        elif x == "linear_3":
            model.linear_3.lora.load_state_dict(lora_state_dict["linear_3"])
    return model


def fine_tune_model(
    model: SimpleNeuralNet, layers: list, rank: int, alpha: int, train_loader: None
):
    if train_loader == None:
        return ValueError("Provide dataset for fine tuning..")

    ##disable_gradient
    for name, p in model.named_parameters():
        p.requires_grad = False

    model = adapt_lora_layers(model=model, layers=layers, rank=rank, alpha=alpha)
    train(train_loader=train_loader, model=model, epochs=5)
    save_lora_weights(model=model, layers=layers, lora_filename=lora_filename)
    return model


def test_fine_tuned_model(test_dataset, layers: list, rank: int, alpha: int):
    ##load pre-trained model
    model = load_model(model_filename=model_filename)

    ##test on existing model
    test(
        test_loader=test_dataset, model=model
    )  ## will provide the same result before fine-tuning

    ##adapt the pre-trained model with lora layers
    model = adapt_lora_layers(model=model, layers=layers, rank=rank, alpha=alpha)

    ##test just after adaptation
    test(
        test_loader=test_dataset, model=model
    )  ## will provide the same result as above since B*A = 0 as A was initialized to zero.

    ##load and merge lora weights with the above model
    model = load_lora_weights(model=model, layers=layers, lora_filename=lora_filename)

    ##test after merge lora weights
    test(
        test_loader=test_dataset, model=model
    )  ## will provide the improved results for digits 8 and 9 as the lora weights are fine-tuned on 8 and 9.


if __name__ == "__main__":
    rank, alpha = 1, 1
    layers = ["linear_1", "linear_2"]
    model = load_model(model_filename=model_filename)

    ##for training:
    # train_loader = get_ds_for_finetuning()

    # model = fine_tune_model(
    #     model=model, layers=layers, rank=rank, alpha=alpha, train_loader=train_loader
    # )

    ## for testing
    test_fine_tuned_model(
        test_dataset=test_loader, layers=layers, rank=rank, alpha=alpha
    )
