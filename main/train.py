import torch
import torchvision
import torch.optim as optim
import utils
import os
import kornia
import psnr


def train_model(net, train_data, val_data, num_epochs, device, train_id):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    net.train()
    LR = 1e-4
    model_folder = "./model/" + train_id
    optimizer = optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999))

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        for batch_idx, sample_batched in enumerate(train_data):
            in_hsv = sample_batched["image"]
            _, _, in_v = torch.split(in_hsv, 1, dim=1)

            S = in_v.to(device)
            S_1 = utils.disturbance(in_v).to(device)

            optimizer.zero_grad()

            R, I = net(S)
            R_1, I_1 = net(S_1)

            loss = utils.calc_loss(R, R_1, I, I_1, S, device)

            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 50 == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(sample_batched["image"]), len(train_data.dataset),
                               100. * batch_idx / len(train_data), loss.item()))

        with torch.no_grad():
            if epoch % 20 == 0:
                net.eval()

                out_path = "./data/result/" + train_id + "/" + str(epoch) + "/img/"
                if not os.path.exists(out_path):
                    os.makedirs(out_path)

                print("========validating========")

                for i, test_batched in enumerate(val_data):
                    index = test_batched["index"][0]
                    in_hsv = test_batched["image"].to(device)
                    h, s, in_v = torch.split(in_hsv, 1, dim=1)

                    out_v, _ = net(in_v)
                    out_hsv = torch.cat((h, s, out_v), dim=1)
                    out_rgb = kornia.color.hsv_to_rgb(out_hsv)

                    img_path = out_path + str(index) + ".JPG"
                    torchvision.utils.save_image(out_rgb, img_path)

                PSNR = psnr.calc_averger_psnr(out_path)
                model_dir = model_folder + "/" + str(epoch) + "(" + str(PSNR) + ").pth"
                torch.save(net.state_dict(), model_dir)
                net.train()
        print('-' * 10)

    print("Training is over.")

    return net
