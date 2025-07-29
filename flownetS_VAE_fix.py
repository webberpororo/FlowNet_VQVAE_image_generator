import torch
from torch import nn, optim
from torch.nn import init
from torch.nn import functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
total_start =time.time()

def conv(batch_norm, in_channels, out_channels, kernel_size=3, stride=1, DSC = False):
    """Convolutional layer with optional batch normalization."""
    layers_list = []
    # if stride != 1:
    #     total_padding = ((16 - 1) * stride + kernel_size - 32)
    #     padding = total_padding // 2
    # else:
    #     padding = (kernel_size - 1) // 2
    padding = (kernel_size - 1) // 2
    # print(padding)
    # padding_left = total_padding // 2
    # padding_right = total_padding - padding_left
    # padding = (padding_left, padding_right, padding_left, padding_right)
    if DSC:
        layers_list.append(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels))
        layers_list.append((nn.Conv2d(in_channels, out_channels, kernel_size=1)))
    else:
        layers_list.append(nn.Conv2d(in_channels, out_channels,
                                 kernel_size=kernel_size, stride = stride, padding=padding))
    if batch_norm:
        layers_list.append(nn.BatchNorm2d(out_channels))
    layers_list.append(nn.ReLU())
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

class Encoder(nn.Module):
    def __init__(self, input_channels = 12, batchNorm=True):
        super().__init__()
        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, input_channels * 2, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 130, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv(self.batchNorm, 258, 512, stride=2)
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
        # print(x.shape)
        x1 = x
        x2 = x
        inputs = torch.cat([x1, x2], dim = 1)
        # print(inputs.shape)
        out_conv1 = self.conv1(inputs)
        # print(out_conv1.shape)

        out_conv2 = self.conv2(out_conv1)
        flow2 = self.predict_flow2(out_conv2)
        out_conv2 = torch.cat((out_conv2, flow2), 1)
        # print(out_conv2.shape)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        flow3 = self.predict_flow3(out_conv3)
        out_conv3 = torch.cat((out_conv3, flow3), 1)
        # print(out_conv3.shape)
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        flow4 = self.predict_flow4(out_conv4)
        out_conv4 = torch.cat((out_conv4, flow4), 1)
        # print(out_conv4.shape)
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        flow5 = self.predict_flow5(out_conv5)
        out_conv5 = torch.cat((out_conv5, flow5), 1)
        # print(out_conv5.shape)
        out_conv6 = self.conv6_1(self.conv6(out_conv5))
        # flow6 = self.predict_flow6(out_conv6)
        # out_conv6 = torch.cat((out_conv6, flow6), 1)
        # print(out_conv6.shape)
        return out_conv2, out_conv3, out_conv4, out_conv5, out_conv6

class Decder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1030, 512)
        self.deconv3 = deconv(1028, 256)
        self.deconv2 = deconv(516, 128)
        self.deconv1 = deconv(260, 64)

        self.de_flow6 = de_flow(1024)
        self.de_flow5 = de_flow(516)
        self.de_flow4 = de_flow(514)
        self.de_flow3 = de_flow(258)
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
        down_flow6 = self.predict_flow6(down_flow6)
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
        #隨機取值呈均勻分布
        if use_ema:
            self.register_buffer("e_i_ts", e_i_ts)
        else:
            self.register_parameter("e_i_ts", nn.Parameter(e_i_ts))
            #上下差在是否為可學習參數
        self.N_i_ts = SonnetExponentialMovingAverage(decay, (num_embeddings,))
        self.m_i_ts = SonnetExponentialMovingAverage(decay, e_i_ts.shape)

    def forward(self, x):
        flat_x = x.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)
        distances = (
            (flat_x ** 2).sum(1, keepdim = True)
            - 2 * flat_x @ self.e_i_ts
            + (self.e_i_ts ** 2).sum(0, keepdims = True)
        )
        distances += torch.randn_like(distances) * distances.std()
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
                print(f"最小 N_i_ts: {self.N_i_ts.average.min().item()} | 最大 N_i_ts: {self.N_i_ts.average.max().item()}")
                unused_codes = self.N_i_ts.average < 1e-2
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
        self.encoder = Encoder(in_channels, True,)
        self.pre_vq_conv = nn.Conv2d(
            in_channels = num_hiddens, out_channels = embedding_dim, kernel_size = 1
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
        # print(dictionary_loss)
        # print(commitment_loss)
        # print(x_recon.shape)
        return {
            "dictionary_loss": dictionary_loss,
            "commitment_loss": commitment_loss,
            "x_recon": x_recon,
        }

if __name__ == '__main__':
    torch.set_printoptions(linewidth = 160)
    def save_img_tensors_as_grid(img_tensors, nrows, f):
        img_tensors = img_tensors.permute(0, 2, 3, 1)
        imgs_array = img_tensors.detach().cpu().numpy()

        #處理向素
        imgs_array[imgs_array < -0.5] = -0.5
        imgs_array[imgs_array > 0.5] = 0.5
        imgs_array = 255 * (imgs_array + 0.5)

        (batch_size, img_size) = img_tensors.shape[:2]
        ncols = batch_size // nrows
        # print(batch_size)
        # print(nrows)
        # print(ncols)
        img_arr = np.zeros((nrows * img_size, ncols * img_size, 3))
        #小圖變大圖
        for idx in range(batch_size):
            row_idx = idx // ncols
            col_idx = idx % ncols
            row_start = row_idx * img_size
            row_end = row_start + img_size
            col_start = col_idx * img_size
            col_end = col_start + img_size
            # print(f"row_start: {row_start}, row_end: {row_end}")
            # print(f"col_start: {col_start}, col_end: {col_end}")
            # print(imgs_array[idx].shape)
            # print(img_arr[row_start:row_end, col_start:col_end].shape)

            img_arr[row_start:row_end, col_start:col_end] = imgs_array[idx]
        # Image.fromarray(img_arr.astype(np.uint8), "RGB").save(f"{f}.jpg")
        plt.imshow(img_arr.astype(np.uint8))
        plt.axis("off")
        plt.show()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_ema = True
    model_args = {
        "in_channels": 3,
        "num_hiddens": 1024,
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
    print(sum(p.numel() for p in model.parameters()))
    # total_num = sum(p.numel() for p in model.parameters())
    # trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(total_num)
    # print(trainable_num)
    # ** 解包字典或鑑值對
    batch_size = 32
    workers = 3
    normalize = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [1.0, 1.0, 1.0])
    #將數據轉換為高斯分布
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    data_root = "./data"
    # train_dataset = CIFAR10(data_root, True, transform, download = True)
    # valid_dataset = CIFAR10(data_root, False, transform, download = True)
    # train_variance = torch.var(torch.tensor(train_dataset.data / 255.0))
    # valid_variance = torch.var(torch.tensor(valid_dataset.data / 255.0))

    # dataset = CIFAR10(data_root, True, transform, download=True)
    # train_size = int(0.8 * len(dataset))
    # valid_size = len(dataset) - train_size
    # train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    # # train_dataset, valid_dataset = train_test_split(dataset, test_size = 0.2, random_state = 42)
    # train_variance = torch.var(torch.tensor(dataset.data[:train_size] / 255.0))
    # valid_variance = torch.var(torch.tensor(dataset.data[train_size:] / 255.0))

    dataset = CIFAR10(data_root, train=True, transform=transform, download=True)

    # 獲取所有索引，然後打亂
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)

    # 計算訓練集和驗證集大小
    train_size = int(0.8 * len(dataset))
    train_indices, valid_indices = indices[:train_size], indices[train_size:]

    # 使用 Subset 創建訓練集和驗證集
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)

    # 取出對應的影像數據
    train_data = dataset.data[train_indices]
    valid_data = dataset.data[valid_indices]

    # 計算方差
    train_variance = torch.var(torch.tensor(train_data, dtype=torch.float32))
    valid_variance = torch.var(torch.tensor(valid_data, dtype=torch.float32))

    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = workers
    )
    valid_lodaer = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        num_workers=workers,
    )
    # num_worker分批工作
    l1_lambda = 1e-7
    l2_lambda = 1e-7
    regulation_type = 'l2'
    beta = 0.01
    train_params = [params for params in model.parameters()]
    lr = 1e-5
    optimizer = optim.Adam(train_params, lr=lr, eps=1e-6)
    criterion = nn.SmoothL1Loss()
    # criterion = nn.MSELoss()
    epochs = 7
    train_every = len(train_loader)
    test_every = len(valid_lodaer)
    best_train_loss = float("inf")

    train_loss = []
    test_loss = []
    train_recon = []
    test_recon = []
    for epoch in range(epochs):
        total_train_loss = 0
        total_recon_error = 0
        n_train = 0
        for (batch_idx, train_tensors) in enumerate(train_loader):
            model.train()
            print(train_tensors[0].shape)
            imgs = train_tensors[0].to(device)
            out = model(imgs)

            if regulation_type == 'l1':
                recon_error = F.l1_loss(out["x_recon"], imgs)
            elif regulation_type == 'l2':
                recon_error = F.mse_loss(out["x_recon"], imgs)
            else:
                recon_error = criterion(out["x_recon"], imgs)
            total_recon_error += recon_error.item()
            loss = recon_error + beta * out["commitment_loss"]
            if not use_ema:
                loss += out["dictionary_loss"]
            total_train_loss += loss.item()

            # recon_error = criterion(out["x_recon"], imgs) / train_variance
            # total_recon_error += recon_error.item()
            # loss = recon_error + beta * out["commitment_loss"]
            # if not use_ema:
            #     loss += out["dictionary_loss"]
            # if regulation_type == 'l1':
            #     l1_loss = sum(p.abs().sum() for p in model.parameters() if p.requires_grad and len(p.shape) > 1)
            #     loss += l1_lambda * l1_loss
            # elif regulation_type == 'l2':
            #     l2_loss = sum((p ** 2).sum() for p in model.parameters() if p.requires_grad and len(p.shape) > 1)
            #     loss += l2_lambda * l2_loss

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            total_norm = torch.norm(
                torch.stack([torch.norm(p.grad, 2) for p in model.parameters() if p.grad is not None]), 2)
                # total_norm = max([torch.max(p.grad.abs()) for p in model.parameters() if p.grad is not None])
            print('total_norm', total_norm)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 防止梯度爆炸
            optimizer.step()
            optimizer.zero_grad()
            n_train += 1
            if ((batch_idx + 1) % train_every) == 0:
                avg_train_loss = total_train_loss / train_every
                avg_recon_error = total_recon_error / train_every

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
                # print(valid_tensors[0].shape)
                imgs = valid_tensors[0].to(device)
                out = model(imgs)
                if regulation_type == 'l1':
                    recon_test_error = F.l1_loss(out["x_recon"], imgs)
                elif regulation_type == 'l2':
                    recon_test_error = F.mse_loss(out["x_recon"], imgs)
                else:
                    recon_test_error = criterion(out["x_recon"], imgs)

                loss = recon_test_error + beta * out["commitment_loss"]
                if not use_ema:
                    loss += out["dictionary_loss"]

                total_recon_test_error += recon_test_error.item()
                total_test_loss += loss.item()
                n_test += 1
                if ((batch_idxs + 1) % test_every) == 0:

                    avg_test_loss = total_test_loss / test_every
                    avg_recon_error_test = total_recon_test_error / test_every

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
                        save_img_tensors_as_grid(valid_tensors[0], 4, "true")
                    save_img_tensors_as_grid(model(valid_tensors[0].to(device))["x_recon"], 4, "recon")

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