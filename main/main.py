import torch
import load_data
import network
import train
import test
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_dir = "./data/SICE_train"
val_dir = "./data/SICE_val/low"
train_id = "MyModel"

train_batch_size = 8
val_batch_size = 1
num_epochs = 500


def main():
    model_folder = "./model/"+train_id

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    model_dir = model_folder + "/final.pth"

    train_data = load_data.load_images(train_dir, train_batch_size)
    val_data = load_data.load_images(val_dir, val_batch_size)

    net = network.UNet(1, 1).to(device)
    net.apply(network.init)

    net = train.train_model(net, train_data, val_data, num_epochs, device, train_id)

    torch.save(net.state_dict(), model_dir)

    test.evaluate(model_dir)


if __name__ == "__main__":
    main()
