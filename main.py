import torch
import torchvision
from PIL import Image
from DICNet import DICNet
import matplotlib.pyplot as plt

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DICNet().to(device)
    net.load_state_dict(torch.load('pre-trained-models.pth',map_location=device))

    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))


    net.eval()
    with torch.no_grad():

        ref_image = Image.open('img_ref.png').convert('L')
        tar_image = Image.open('img_tar.png').convert('L')
        ref_image = torchvision.transforms.ToTensor()(ref_image)
        tar_image = torchvision.transforms.ToTensor()(tar_image)
        image = torch.cat((ref_image, tar_image), dim=0).unsqueeze(dim=0).to(device)
        image = torchvision.transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5])(image)

        dis = net(image).cpu()*15
       

        plt.subplot(121)
        plt.imshow(dis[0][0].numpy(),cmap='jet')
        plt.subplot(122)
        plt.imshow(dis[0][1].numpy(),cmap='jet')
        plt.savefig("res.jpg")
        