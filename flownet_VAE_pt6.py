import torch
from torch import nn, optim
from torch.nn import init
from torch.nn import functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torchvision import transforms, datasets
from torchvision.transforms import InterpolationMode
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import time
import os

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

total_start = time.time()

from sklearn.model_selection import train_test_split

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# print(torch.version.cuda)  # PyTorch 使用的 CUDA 版本
# print(torch.cuda.is_available())  # 檢查是否有 CUDA
# print(torch.backends.cudnn.version())  # cuDNN 版本

# print(torch.__version__)
# print(torch.version.cuda)

# 定義點積操作
class MyDot(nn.Module):
    def forward(self, x1, x2):
        # x1 和 x2 的最後一個維度進行逐元素相乘後求和
        return torch.sum(x1 * x2, dim=-1, keepdim=True)


# 定義裁剪和填充操作
def get_padded_stride(feature_map, displacement_x, displacement_y, height_8, width_8):
    slice_height = height_8 - abs(displacement_y)
    slice_width = width_8 - abs(displacement_x)
    start_y = abs(displacement_y) if displacement_y < 0 else 0
    start_x = abs(displacement_x) if displacement_x < 0 else 0
    top_pad = displacement_y if displacement_y > 0 else 0
    bottom_pad = start_y
    left_pad = displacement_x if displacement_x > 0 else 0
    right_pad = start_x

    # 使用 PyTorch 的裁剪和填充
    sliced = feature_map[:, :, start_y:start_y + slice_height, start_x:start_x + slice_width]
    padded = F.pad(sliced, (left_pad, right_pad, top_pad, bottom_pad))
    return padded


# 定義相關性層
class CorrelationLayer(nn.Module):
    def __init__(self, max_displacement=20, stride2=2, height_8=384 // 8, width_8=512 // 8):
        super(CorrelationLayer, self).__init__()
        self.max_displacement = max_displacement
        self.stride2 = stride2
        self.height_8 = height_8
        self.width_8 = width_8
        self.dot_layer = MyDot()

    def forward(self, conv3_pool_l, conv3_pool_r):
        devices = torch.device('cpu')
        # print(conv3_pool_l.shape)
        self.height_8 = conv3_pool_l.shape[2]
        self.width_8 = conv3_pool_l.shape[3]
        conv3_pool_l = conv3_pool_l.permute(0, 2, 3, 1)

        layer_list = []
        for i in range(-self.max_displacement, self.max_displacement + self.stride2, self.stride2):
            for j in range(-self.max_displacement, self.max_displacement + self.stride2, self.stride2):
                slice_b = get_padded_stride(conv3_pool_r, i, j, self.height_8, self.width_8)
                slice_b = slice_b.permute(0, 2, 3, 1)

                current_layer = self.dot_layer(conv3_pool_l, slice_b)
                current_layer = current_layer.permute(0, 3, 1, 2).to(devices)
                layer_list.append(current_layer)

        # 將所有偏移的結果在通道維度拼接
        for i, t in enumerate(layer_list):
            if torch.isnan(t).any() or torch.isinf(t).any():
                print(f"Layer {i} has NaN or Inf!")
        torch.cuda.synchronize()

        outs = torch.cat(layer_list, dim=1)
        outs = outs.to(device)
        return outs

# 在所有層添加 hook
def check_forward_hook(module, input, output):
    if torch.isinf(output).any():
        print(f"🚨 WARNING: Inf detected in {module}")
        print(f"Input stats: min={input[0].min()}, max={input[0].max()}")
        print(f"Output stats: min={output.min()}, max={output.max()}")
    elif torch.isnan(output).any():
        print(f"🚨 WARNING: NaN detected in {module}")
        print(f"Input stats: min={input[0].min()}, max={input[0].max()}")
        print(f"Output stats: min={output.min()}, max={output.max()}")

def conv(batch_norm, in_channels, out_channels, kernel_size=3, stride=1, DSC=False):
    """Convolutional layer with optional batch normalization."""
    layers_list = []
    padding = (kernel_size - 1) // 2
    if DSC:
        layers_list.append(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=in_channels))
        layers_list.append((nn.Conv2d(in_channels, out_channels, kernel_size=1)))
    else:
        layers_list.append(nn.Conv2d(in_channels, out_channels,
                                     kernel_size=kernel_size, stride=stride, padding=padding))
    if batch_norm:
        layers_list.append(nn.BatchNorm2d(out_channels))
    layers_list.append(nn.ReLU())

    # 在所有層添加 hook
    for layer in layers_list:
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.ReLU):
            layer.register_forward_hook(check_forward_hook)

    return nn.Sequential(*layers_list)


def predict_flow(in_channels):
    """Predict flow output layer."""
    return nn.Conv2d(in_channels, 2, kernel_size=3, padding=1)


def deconv(in_channels, out_channels):
    """Deconvolutional layer."""
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

def de_flow(in_channels):
    """Deconvolutional layer."""
    return nn.ConvTranspose2d(in_channels, 2, kernel_size=4, stride=2, padding=1)

def crop_like(tensor1, tensor2):
    """Crop tensor1 to the size of tensor2 using bilinear interpolation."""
    target_size = tensor2.size()[2:]  # (height, width) of tensor2
    return F.interpolate(tensor1, size=target_size, mode='bilinear', align_corners=False)
    # 功能類似上採樣，讓影像變清晰,bilinear是雙線性插值,align_corners是確保計算更精細


class tofp32(torch.nn.Module):
    def forward(self, x):
        return x.float()  # 將張量轉換為 FP32


class tofp16(torch.nn.Module):
    def forward(self, x):
        return x.half()  # 將張量轉換為 FP16


class Encoder(nn.Module):
    def __init__(self, input_channels = 12, batchNorm=True, div_flow = 20, fp16 = False):
        super().__init__()
        self.batchNorm = batchNorm
        # self.div_flow = div_flow

        self.conv1 = conv(self.batchNorm, 3, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 130, 256, kernel_size=5, stride=2)
        self.conv_redir = conv(self.batchNorm, 258, 32, kernel_size=1, stride=1)

        if fp16:
            self.corr = nn.Sequential(
                tofp32(),
                CorrelationLayer(),
                tofp16())
        else:
            self.corr = CorrelationLayer()
        # self.down_conv = nn.Conv2d(256 * 441, 441, kernel_size=1)

        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)
        self.conv3_1 = conv(self.batchNorm, 473, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 514, 514, stride=2)
        self.conv5_1 = conv(self.batchNorm, 514, 514)
        self.conv6 = conv(self.batchNorm, 516, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(514)
        self.predict_flow4 = predict_flow(512)
        self.predict_flow3 = predict_flow(256)
        self.predict_flow2 = predict_flow(128)

        for m in self.modules(): # 幫助NN初始化
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)

    def forward(self, x):
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3::, :, :]
        # x1 = x
        # x2 = x

        out_conv1a = self.conv1(x1)

        out_conv2a = self.conv2(out_conv1a)
        flow2a = self.predict_flow2(out_conv2a)
        out_conv2a = torch.cat((out_conv2a, flow2a), 1)

        out_conv3a = self.conv3(out_conv2a)
        flow3a = self.predict_flow3(out_conv3a)
        out_conv3a = torch.cat((out_conv3a, flow3a), 1)

        # FlownetC bottom input stream
        out_conv1b = self.conv1(x2)

        out_conv2b = self.conv2(out_conv1b)
        flow2b = self.predict_flow2(out_conv2b)
        out_conv2b = torch.cat((out_conv2b, flow2b), 1)

        out_conv3b = self.conv3(out_conv2b)
        flow3b = self.predict_flow3(out_conv3b)
        out_conv3b = torch.cat((out_conv3b, flow3b), 1)

        # Merge streams
        # print(out_conv3a.shape)
        start2 = time.time()
        out_corr = self.corr(out_conv3a, out_conv3b)  # False
        end2 = time.time()
        print('corr_time', end2 - start2)
        # print(out_corr.shape)
        # out_corr = self.down_conv(out_corr)
        out_corr = self.corr_activation(out_corr)

        # Redirect top input stream and concatenate
        out_conv_redir = self.conv_redir(out_conv3a)
        # print(out_corr.shape)
        # print(out_conv_redir.shape)

        in_conv3_1 = torch.cat((out_conv_redir, out_corr), 1)

        # Merged conv layers
        out_conv3_1 = self.conv3_1(in_conv3_1)

        out_conv4 = self.conv4_1(self.conv4(out_conv3_1))
        flow4 = self.predict_flow4(out_conv4)
        out_conv4 = torch.cat((out_conv4, flow4), 1)

        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        flow5 = self.predict_flow5(out_conv5)
        out_conv5 = torch.cat((out_conv5, flow5), 1)

        out_conv6 = self.conv6_1(self.conv6(out_conv5))
        flow6 = self.predict_flow6(out_conv6)
        out_conv6 = torch.cat((out_conv6, flow6), 1)

        # print(out_conv3_1.shape)
        # print(out_conv4.shape)
        # print(out_conv5.shape)
        # print(out_conv6.shape)

        return out_conv2a, out_conv3_1, out_conv4, out_conv5, out_conv6

class Decder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv5 = deconv(1026, 512)
        self.deconv4 = deconv(1030, 512)
        self.deconv3 = deconv(1028, 256)
        self.deconv2 = deconv(514, 128)
        self.deconv1 = deconv(260, 64)

        self.de_flow6 = de_flow(1026)
        self.de_flow5 = de_flow(516)
        self.de_flow4 = de_flow(514)
        self.de_flow3 = de_flow(256)
        self.de_flow2 = de_flow(130)

        self.predict_flow6 = nn.Conv2d(2, 2, 4, 2, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)
        self.small_down = nn.Conv2d(66, 3, kernel_size=1, stride=1, padding=0)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
    def forward(self, x):
        out_conv2 = x[0]
        out_conv3 = x[1]
        out_conv4 = x[2]
        out_conv5 = x[3]
        out_conv6 = x[4]

        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)
        down_flow6 = self.de_flow6(out_conv6)
        out_deconv5 = torch.cat((out_deconv5, out_conv5, down_flow6), 1)

        out_deconv4 = crop_like(self.deconv4(out_deconv5), out_conv4)
        down_flow5 = self.de_flow5(out_conv5)
        out_deconv4 = torch.cat((out_deconv4, out_conv4, down_flow5), 1)

        out_deconv3 = crop_like(self.deconv3(out_deconv4), out_conv3)
        down_flow4 = self.de_flow4(out_conv4)
        out_deconv3 = torch.cat((out_deconv3, out_conv3, down_flow4), 1)

        out_deconv2 = crop_like(self.deconv2(out_deconv3), out_conv2)
        down_flow3 = self.de_flow3(out_conv3)
        out_deconv2 = torch.cat((out_deconv2, out_conv2, down_flow3), 1)

        out_deconv1 = self.deconv1(out_deconv2)
        down_flow2 = self.de_flow2(out_conv2)
        out_deconv1 = torch.cat((out_deconv1, down_flow2), 1)

        flow2 = self.small_down(out_deconv1)

        final = self.upsample1(flow2)

        return final


class SonnetExponentialMovingAverage(nn.Module):
    def __init__(self, decay, shape):
        super().__init__()
        self.decay = decay
        self.counter = 0
        self.register_buffer("hidden", torch.zeros(*shape))
        # 註冊持久性(persistent)張量，不會出現在model.parameters() 或 optimizer
        # * 是Python 中的解包語法（unpacking），將每個維度解包為獨立參數(解包可迭代參數)
        self.register_buffer("average", torch.zeros(*shape))

    def update(self, value):
        self.counter += 1
        with torch.no_grad():
            self.hidden -= (self.hidden - value) * (1 - self.decay)
            self.average = self.hidden / (1 - self.decay ** self.counter)

    def __call__(self, value):
        self.update(value)
        return self.average


class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, use_ema, decay, epsilon):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.use_ema = use_ema
        self.decay = decay
        self.epsilon = epsilon
        limit = 3 ** 0.5
        e_i_ts = torch.FloatTensor(embedding_dim, num_embeddings).uniform_(-limit, limit)
        # 隨機取值呈均勻分布
        if use_ema:
            self.register_buffer("e_i_ts", e_i_ts)
        else:
            self.register_parameter("e_i_ts", nn.Parameter(e_i_ts))
            # 上下差在是否為可學習參數
        self.N_i_ts = SonnetExponentialMovingAverage(decay, (num_embeddings,))
        self.m_i_ts = SonnetExponentialMovingAverage(decay, e_i_ts.shape)

    def forward(self, x):
        flat_x = x.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)
        distances = (
                (flat_x ** 2).sum(1, keepdim=True)
                - 2 * flat_x @ self.e_i_ts
                + (self.e_i_ts ** 2).sum(0, keepdims=True)
        )
        distances += torch.randn_like(distances) * distances.std()

        if torch.isnan(self.e_i_ts).any():
            print("NaN/inf detected in e_i_ts!")
        if torch.isnan(flat_x).any():
            print("NaN/inf detected in flat_x!")
        if torch.isnan(distances).any():
            print("NaN/inf detected in distances!")
        if torch.isinf(self.e_i_ts).any():
            print("inf detected in e_i_ts!")
        if torch.isinf(flat_x).any():
            print("inf detected in flat_x!")
        if torch.isinf(distances).any():
            print("inf detected in distances!")

        # @ 矩陣乘法
        encoding_indices = distances.argmin(1)
        quantized_x = F.embedding(
            encoding_indices.view(x.shape[0], *x.shape[2:]), self.e_i_ts.transpose(0, 1)
        ).permute(0, 3, 1, 2)
        if not self.use_ema:
            dictionary_loss = ((x.detach() - quantized_x) ** 2).mean()
            # .detach()與原始張量共享數據，但之後不參與任何計算圖，也就是從計算圖中剝離出來
        else:
            dictionary_loss = None
        commitment_loss = ((x - quantized_x.detach()) ** 2).mean()
        quantized_x = x + (quantized_x - x).detach()
        if self.use_ema and self.training:
            with torch.no_grad():
                encoding_one_hots = F.one_hot(
                    encoding_indices, self.num_embeddings
                ).type(flat_x.dtype)
                n_i_ts = encoding_one_hots.sum(0)
                self.N_i_ts(n_i_ts)
                self.N_i_ts.hidden = torch.clamp(self.N_i_ts.hidden + 1e-3, min=1e-3, max=1e5)
                # self.N_i_ts.hidden = self.N_i_ts.hidden.clamp_min(1e-3)
                embed_sums = flat_x.transpose(0, 1) @ encoding_one_hots
                self.m_i_ts(embed_sums)
                N_i_ts_sum = self.N_i_ts.average.sum()
                N_i_ts_stable = (
                        (self.N_i_ts.average + self.epsilon)
                        / (N_i_ts_sum + self.num_embeddings * self.epsilon).clamp_min(1e-3)
                        * N_i_ts_sum
                )
                N_i_ts_stable = torch.clamp(N_i_ts_stable, min=self.epsilon)
                if torch.isnan(self.m_i_ts.average).any():
                    print("NaN detected in m_i_ts.average before update")
                print(f"最小 N_i_ts: {self.N_i_ts.average.min().item()} | 最大 N_i_ts: {self.N_i_ts.average.max().item()}")
                unused_codes = self.N_i_ts.average < 1e-2  # 找出長時間沒被選中的嵌入
                if unused_codes.any():
                    self.e_i_ts[:, unused_codes] = torch.randn_like(self.e_i_ts[:, unused_codes]) * 0.1
                self.e_i_ts = self.m_i_ts.average / N_i_ts_stable.unsqueeze(0)

        return (
            quantized_x,
            dictionary_loss,
            commitment_loss,
            encoding_indices.view(x.shape[0], -1),
        )


class VQVAE(nn.Module):
    def __init__(self,
                 in_channels,
                 num_hiddens,
                 num_downsampling_layers,
                 num_residual_layers,
                 num_residual_hiddens,
                 embedding_dim,
                 num_embeddings,
                 use_ema,
                 decay,
                 epsilon,
                 ):
        super().__init__()
        self.encoder = Encoder(in_channels, True, )
        self.pre_vq_conv = nn.Conv2d(
            in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1
        )
        self.vq = VectorQuantizer(
            embedding_dim, num_embeddings, use_ema, decay, epsilon
        )
        self.decoder = Decder()

    def quantize(self, x):
        output = self.encoder(x)
        outs = output[4]

        z = self.pre_vq_conv(outs)
        (z_quantized, dictionary_loss, commitment_loss, encoding_indices) = self.vq(z)

        return (output, dictionary_loss, commitment_loss, encoding_indices)

    def forward(self, x):
        (z_quantized, dictionary_loss, commitment_loss, encoding_indices) = self.quantize(x)
        x_recon = self.decoder(z_quantized)
        return {
            "dictionary_loss": dictionary_loss,
            "commitment_loss": commitment_loss,
            "x_recon": x_recon,
        }

if __name__ == '__main__':
    torch.set_printoptions(linewidth=160)


    def save_img_tensors_as_grid(img_tensors, nrows, f):
        target_size = 32
        img_tensors = F.interpolate(img_tensors, size=(target_size, target_size), mode='bilinear', align_corners=False)
        img_tensors = img_tensors.permute(0, 2, 3, 1)
        imgs_array = img_tensors.detach().cpu().numpy()

        # 處理向素
        imgs_array[imgs_array < -0.5] = -0.5
        imgs_array[imgs_array > 0.5] = 0.5
        imgs_array = 255 * (imgs_array + 0.5)

        (batch_size, img_size) = img_tensors.shape[:2]
        ncols = batch_size // nrows
        img_arr = np.zeros((nrows * batch_size, ncols * batch_size, 3))
        # 小圖變大圖
        for idx in range(batch_size):
            row_idx = idx // ncols
            col_idx = idx % ncols
            row_start = row_idx * img_size
            row_end = row_start + img_size
            col_start = col_idx * img_size
            col_end = col_start + img_size

            img_arr[row_start:row_end, col_start:col_end] = imgs_array[idx]
        plt.imshow(img_arr.astype(np.uint8))
        plt.axis("off")
        plt.show()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_ema = True
    model_args = {
        "in_channels": 3,
        "num_hiddens": 1026,
        "num_downsampling_layers": 2,
        "num_residual_layers": 2,
        "num_residual_hiddens": 32,
        "embedding_dim": 32,
        "num_embeddings": 128,
        "use_ema": use_ema,
        "decay": 0.99,
        "epsilon": 1e-3,
    }
    model = VQVAE(**model_args).to(device)
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_num)
    print(trainable_num)
    # ** 解包字典或鑑值對
    batch_size = 32
    workers = 0
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
    # 將數據轉換為高斯分布
    transform = transforms.Compose(
        [
            transforms.Resize((192, 192), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            normalize,
        ]
    )
    data_root = "./data"
    # train_dataset = CIFAR10(data_root, True, transform, download = True)
    # valid_dataset = CIFAR10(data_root, False, transform, download = True)

    # dataset = CIFAR10(data_root, True, transform, download=True)
    # train_size = int(0.8 * len(dataset))
    # valid_size = len(dataset) - train_size
    # train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    # # train_dataset, valid_dataset = train_test_split(dataset, test_size = 0.2, random_state = 42)
    # train_variance = torch.var(torch.tensor(dataset.data[:train_size] / 255.0))
    # valid_variance = torch.var(torch.tensor(dataset.data[train_size:] / 255.0))

    # np.save("train_data.npy", train_dataset.data)
    # np.save("valid_data.npy", valid_dataset.data)
    # train_dataset = np.load("train_data.npy")
    # valid_dataset = np.load("valid_data.npy")
    # train_dataset, valid_dataset = train_test_split(dataset, test_size = 0.2, random_state = 42)

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
    # ])
    #
    # train_dataset = datasets.ImageFolder(root="D:\\archive\\afhq\\train", transform=transform)
    # valid_dataset = datasets.ImageFolder(root="D:\\archive\\afhq\\val", transform=transform)
    #
    # # print(train_dataset.class_to_idx)  # {'class1': 0, 'class2': 1}

    class PairedImageDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.image_files = sorted([f for f in os.listdir(root_dir) if f.endswith(".png")])
            self.transform = transform

            # 预先分组，确保相邻两张图片为一组
            self.paired_images = [(self.image_files[i], self.image_files[i + 1])
                                  for i in range(0, len(self.image_files), 2)]

        def __len__(self):
            return len(self.paired_images)

        def __getitem__(self, idx):
            img1_path = os.path.join(self.root_dir, self.paired_images[idx][0])
            img2_path = os.path.join(self.root_dir, self.paired_images[idx][1])

            img1 = Image.open(img1_path).convert("RGB")
            # print('1')
            # img1.show()
            img2 = Image.open(img2_path).convert("RGB")
            # print('2')
            # img2.show()

            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            return img1, img2  # 直接返回一对图片


    dataset = PairedImageDataset(root_dir="D:\\ChairsSDHom_extended\\ChairsSDHom_extended\\train2\\class2\\class1",
                                 transform=transform)
    print(len(dataset))
    # dataset = CIFAR10(data_root, train=True, transform=transform, download=True)

    # 獲取所有索引，然後打亂
    indices = np.arange(len(dataset))
    # np.random.shuffle(indices)

    # 計算訓練集和驗證集大小
    train_size = int(0.8 * len(dataset))
    train_indices, valid_indices = indices[:train_size], indices[train_size:]

    # 使用 Subset 創建訓練集和驗證集
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)
    # print(train_dataset)

    # class PairDataset(Dataset):
    #     def __init__(self, dataset):
    #         """
    #         dataset: 原始的 CIFAR-10 数据集
    #         """
    #         self.dataset = dataset
    #         self.length = len(dataset) // 2  # 每两张作为一组，长度减半
    #
    #     def __len__(self):
    #         return self.length
    #
    #     def __getitem__(self, idx):
    #         """
    #         返回两个连续的图像及其标签
    #         """
    #         img1 = self.dataset[idx * 2]
    #         img2 = self.dataset[idx * 2 + 1]
    #         return [img1, img2]
    #
    #
    # train_dataset = PairDataset(train_dataset)
    # valid_dataset = PairDataset(valid_dataset)
    # # print(len(train_dataset))

    # 取出對應的影像數據
    # train_data = dataset.data[train_indices]
    # valid_data = dataset.data[valid_indices]
    train_data = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    valid_data = torch.stack([valid_dataset[i][0] for i in range(len(valid_dataset))])

    train_variance = torch.var(torch.tensor(train_data, dtype=torch.float32))
    valid_variance = torch.var(torch.tensor(valid_data, dtype=torch.float32))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=False
    )
    valid_lodaer = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=False
    )
    # num_worker分批工作
    l1_lambda = 1e-7
    l2_lambda = 1e-7
    regulation_type = 'l2'
    beta = 0.01
    train_params = [params for params in model.parameters()]
    lr = 1e-5
    optimizer = optim.Adam(train_params, lr=lr, eps = 1e-6)
    criterion = nn.MSELoss()
    epochs = 15
    train_every = len(train_loader)
    test_every = len(valid_lodaer)
    # best_train_loss = float("inf")

    train_loss = []
    test_loss = []
    train_recon = []
    test_recon = []
    start = time.time()
    scaler = torch.cuda.amp.GradScaler()  # 訓練前先初始化AMP的GradScaler
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_recon_error = 0
        n_train = 0
        for (batch_idx, train_tensors) in enumerate(train_loader):
            end = time.time()
            print(batch_idx, end - start)
            start = end

            imgs = train_tensors[0].to(device)
            imgs2 = train_tensors[1].to(device)
            inputs = torch.cat((imgs, imgs2), 1)
            with torch.cuda.amp.autocast():
                optimizer.zero_grad()
                out = model(inputs)

                if regulation_type == 'l1':
                    recon_error = F.l1_loss(out["x_recon"], imgs2)
                elif regulation_type == 'l2':
                    recon_error = F.mse_loss(out["x_recon"], imgs2) / train_variance
                else:
                    recon_error = criterion(out["x_recon"], imgs2) / train_variance
                loss = recon_error + beta * out["commitment_loss"]
                if not use_ema:
                    loss += out["dictionary_loss"]

                if torch.isnan(loss) or torch.isnan(loss):
                    print("NaN detected in loss!")
                if torch.isnan(out["x_recon"]).any():
                    print("NaN detected in model output!")
                if torch.isinf(loss) or torch.isnan(loss):
                    print("inf detected in loss!")
                if torch.isinf(out["x_recon"]).any():
                    print("inf detected in model output!")
                for p in model.parameters():
                    if p.grad is not None and torch.isnan(p.grad).any():
                        print("NaN detected in gradients!")

                # recon_error = criterion(out["x_recon"], imgs) / train_variance
                # loss = recon_error + beta * out["commitment_loss"]
                # if not use_ema:
                #     loss += out["dictionary_loss"]
                # if regulation_type == 'l1':
                #     l1_loss = sum(p.abs().sum() for p in model.parameters() if p.requires_grad and len(p.shape) > 1)
                #     loss += l1_lambda * l1_loss
                # elif regulation_type == 'l2':
                #     l2_loss = sum((p ** 2).sum() for p in model.parameters() if p.requires_grad and len(p.shape) > 1)
                #     loss += l2_lambda * l2_loss

            scaler.scale(loss).backward()  # 缩放loss並反向傳播

            for p in model.parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    print(f"NaN detected in {p.name} gradient!")

            scaler.unscale_(optimizer)  # 反縮放 optimizer
            total_norm = torch.norm(
                torch.stack([torch.norm(p.grad, 2) for p in model.parameters() if p.grad is not None]), 2)
            # total_norm = max([torch.max(p.grad.abs()) for p in model.parameters() if p.grad is not None])
            print('total_norm', total_norm)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1, norm_type=1)
            scaler.step(optimizer)  # 使用缩放後的梯度更新参数
            scaler.update()  # 動態調整 GradScaler，避免梯度過小或溢出

            # loss.backward()
            # optimizer.step()

            total_train_loss += loss.item()
            total_recon_error += recon_error.item()
            n_train += 1

            if ((batch_idx + 1) % train_every) == 0:
                avg_train_loss = total_train_loss / n_train
                avg_recon_error = total_recon_error / n_train

                print(f"epoch: {epoch}\nbatch_idx: {batch_idx + 1}", flush=True)
                print(f"total_train_loss: {avg_train_loss}")
                print(f"recon_error: {avg_recon_error}\n")

                train_loss.append(avg_train_loss)
                train_recon.append(avg_recon_error)
                total_train_loss = 0
                total_recon_error = 0
                n_train = 0

        model.eval()
        with torch.no_grad():

            total_test_loss = 0
            total_recon_test_error = 0
            n_test = 0
            for (batch_idxs, valid_tensors) in enumerate(valid_lodaer):
                print(batch_idxs)
                # print(valid_tensors[0].shape)
                imgs = valid_tensors[0].to(device)
                imgs2 = valid_tensors[1].to(device)
                inputs = torch.cat((imgs, imgs2), 1)
                out = model(inputs)

                if regulation_type == 'l1':
                    recon_test_error = F.l1_loss(out["x_recon"], imgs2)
                elif regulation_type == 'l2':
                    recon_test_error = F.mse_loss(out["x_recon"], imgs2) / valid_variance
                else:
                    recon_test_error = criterion(out["x_recon"], imgs2) / valid_variance

                loss = recon_test_error + beta * out["commitment_loss"]
                if not use_ema:
                    loss += out["dictionary_loss"]

                # recon_test_error = criterion(out["x_recon"], imgs) / valid_variance
                # loss = recon_test_error + beta * out["commitment_loss"]
                # if not use_ema:
                #     loss += out["dictionary_loss"]

                total_recon_test_error += recon_test_error.item()
                total_test_loss += loss.item()
                n_test += 1
                if ((batch_idxs + 1) % test_every) == 0:
                    avg_test_loss = total_test_loss / n_test
                    avg_recon_error_test = total_recon_test_error / n_test

                    print(f"epoch: {epoch}\nbatch_idx: {batch_idxs + 1}", flush=True)
                    print(f"total_train_loss: {avg_test_loss}")
                    print(f"recon_error: {avg_recon_error_test}\n")

                    test_loss.append(avg_test_loss)
                    test_recon.append(avg_recon_error_test)
                    total_test_loss = 0
                    total_recon_test_error = 0
                    n_test = 0
                if n_test == 1:
                    if epoch == 0:
                        print(valid_tensors[0].shape)
                        save_img_tensors_as_grid(valid_tensors[1], 4, "true")
                    save_img_tensors_as_grid(model(inputs.to(device))["x_recon"], 4, "recon")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_loss, label='Train Loss', marker='o')
    plt.plot(range(1, epochs + 1), test_loss, label='Test Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_recon, label='Train recon', marker='o')
    plt.plot(range(1, epochs + 1), test_recon, label='Test recon', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('recon')
    plt.title('recon vs. Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()

    total_end = time.time()
    print('time', total_end - total_start)
    print('train_loss', train_loss)
    print('test_loss', test_loss)
    print('train_recon', train_recon)
    print('test_recon', test_recon)