import torch
import torchvision
import torch.optim
import os
import network
from PIL import Image
import kornia
import glob
import time
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_dir = "./model/MyModel/final.pth"
transform = transforms.Compose([
    transforms.ToTensor()
])

net = network.UNet(1, 1).to(device)
net.load_state_dict(torch.load(model_dir))
net.eval()


def test_model(image_path, model_dir):
    folder_dir = model_dir[15:-4]

    img = Image.open(image_path)
    img = transform(img).to(device)

    st = time.time()
    img_hsv = kornia.color.rgb_to_hsv(img)

    img_hsv = img_hsv.unsqueeze(0)
    h, s, in_v = torch.split(img_hsv, 1, dim=1)

    out_v, _ = net(in_v)

    out_hsv = torch.cat((h, s, out_v), dim=1)
    out_rgb = kornia.color.hsv_to_rgb(out_hsv)

    torch.cuda.synchronize()
    print(time.time() - st)

    out_path = image_path.replace('test/SICE_test/low/', 'result/' + folder_dir + '/SICE/')

    torchvision.utils.save_image(out_rgb, out_path)


def evaluate(model_dir):
    with torch.no_grad():

        filePath = "./data/test/SICE_test/low/"

        folder_dir = model_dir[15:-4]
        file_list = os.listdir(filePath)

        out_path = filePath.replace('test/SICE_test/low/', 'result/' + folder_dir + '/SICE/')
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        for file_name in file_list:
            test_list = glob.glob(filePath + file_name)
            for image in test_list:
                test_model(image, model_dir)


if __name__ == '__main__':
    evaluate(model_dir)
