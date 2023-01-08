import numpy as np
import mlflow.pytorch

import torch
import torchvision

import sklearn
import sklearn.datasets


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return device


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.fc1 = torch.nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x

    def evaluate_accuracy(self, images, labels, batch=4):
        device = get_device()
        
        self.to(device)
        
        running_accuracy = 0
        for index in range(len(images) // batch):
            images_batch = images[index:index+batch]
            labels_batch = labels[index:index+batch]
            
            x = torch.autograd.Variable(
                torch.from_numpy(images_batch).to(device))
            y = torch.autograd.Variable(
                torch.from_numpy(labels_batch).to(device))

            outputs = self(x)

            step_accuracy = torch.sum(
                torch.eq(torch.argmax(outputs, dim=1), y)
            ) / len(y)

            running_accuracy = (
                running_accuracy * index + step_accuracy
            ) / (index + 1)

        return running_accuracy

    def train(self, images, labels, epochs=-1, batch=4):
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch", batch)

        device = get_device()

        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters())

        self.to(device)

        epoch = 0
        while True:
            mean_loss = 0

            for index in range(len(images) // batch):
                images_batch = images[index:index+batch]
                labels_batch = labels[index:index+batch]
                
                x = torch.autograd.Variable(
                    torch.from_numpy(images_batch).to(device))
                y = torch.autograd.Variable(
                    torch.from_numpy(labels_batch).to(device))

                optimizer.zero_grad()
                outputs = self(x)
                loss = loss_func(outputs, y)
                loss.backward()
                optimizer.step()

                mean_loss = (mean_loss * index + loss.item()) / (index + 1)

            accuracy_train = self.evaluate_accuracy(images, labels)
            
            print("Epoch: {} --> loss: {} accuracy(t): {}".format(
                epoch, mean_loss, accuracy_train
            ))

            mlflow.log_metric("loss", mean_loss, step=epoch)
            mlflow.log_metric("accuracy_train", accuracy_train, step=epoch)

            # Stopping criteria
            if epochs > 0 and epoch == epochs:
                break
            elif epochs < 0 and accuracy_train > .99:
                break
            
            epoch += 1

        mlflow.pytorch.log_model(self, "models")


def load_dataset():
    mnist = sklearn.datasets.load_digits()

    images = (
        mnist.data.reshape((-1, 8, 8, 1)) / np.max(mnist.data)
    ).astype(np.float32)
    images = images.transpose((0, 3, 1, 2))

    print(images.shape, np.min(images), np.mean(images), np.max(images))

    labels = mnist.target

    return images, labels


def train(model):
    images, labels = load_dataset()

    model.train(images, labels, epochs=-1)


def main():
    mlflow.start_run()

    model = Model()

    train(model)

    mlflow.end_run()


if __name__ == "__main__":
    main()
