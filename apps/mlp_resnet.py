import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    residual_part = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim),
    )
    return nn.Sequential(nn.Residual(residual_part), nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    mlp_resnet = [
        nn.Linear(dim, hidden_dim),
        nn.ReLU()
    ]
    for i in range(num_blocks):
        mlp_resnet.append(ResidualBlock(
            dim=hidden_dim, 
            hidden_dim=hidden_dim//2, 
            norm=norm, 
            drop_prob=drop_prob
        ))
    mlp_resnet.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*mlp_resnet)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt:
        model.train()
    else:
        model.eval()
    error_tot = 0.
    loss_tot = 0.
    data_num = 0
    for X, y in dataloader:
        y_hat = model.forward(X)
        loss = nn.SoftmaxLoss().forward(y_hat, y)
        if opt:
            opt.reset_grad()
            loss.backward()
            opt.step()

        data_num += X.shape[0]
        loss_tot += (loss.numpy() * X.shape[0])
        error_tot += (np.argmax(y_hat.numpy(), axis=1) != y.numpy()).sum()
    return float(error_tot / data_num), float(loss_tot / data_num)
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_image_filename = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    train_label_filename = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    mnist_train_dataset = ndl.data.MNISTDataset(train_image_filename, train_label_filename)
    mnist_train_dataloader = ndl.data.DataLoader(mnist_train_dataset, batch_size, shuffle=True)
    
    model = MLPResNet(dim=mnist_train_dataset[0][0].shape[1], hidden_dim=hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for e in range(epochs):
        training_loss, training_error = epoch(mnist_train_dataloader, model, opt)

    test_image_filename = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    test_label_filename = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
    mnist_test_dataset = ndl.data.MNISTDataset(test_image_filename, test_label_filename)
    mnist_test_dataloader = ndl.data.DataLoader(mnist_test_dataset, batch_size)
    test_loss, test_error = epoch(mnist_test_dataloader, model)

    return training_loss, training_error, test_loss, test_error
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
