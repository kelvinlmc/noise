from torch import nn
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from PIL import Image


def get_noise(n_samples, z_dim):
    return torch.randn(n_samples,z_dim)
def get_generator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True)
    )
class Generator(nn.Module):
    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        super(Generator, self).__init__()
        # Build the neural network
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()
        )
    def forward(self, noise):
        return self.gen(noise)
    def get_gen(self):
        return self.gen

gen = Generator(64).to('cpu')
#print(type(gen))
fake_noise = get_noise(128, 64)
print("原始：",fake_noise.size())
#ffke=get_generator_block(fake_noise,128)
#print("模块：",ffke.size())
fake = gen(fake_noise)    #这里是传入的一个tensor,误以为是传入参数，这个tensor是跟全连接层相乘的。
print("最后：",fake.size())
fake1=fake[0]
fake1=torch.reshape(fake1,(1,28,28))
#print(fake.size())
#print(fake1.size())
ToIM=transforms.ToPILImage()
image=ToIM(fake1)
image.show()



'''
dataloader = DataLoader(MNIST('.', download=True, transform=transforms.ToTensor()),batch_size=128,shuffle=True)
for real, _ in tqdm(dataloader):
    cur_batch_size = len(real)   #real是batch_size个真实图片
    real = real.view(cur_batch_size, -1)
    print(real)
    print(_)
    print(cur_batch_size)
'''
ten=torch.ones(128,1)
print(ten)