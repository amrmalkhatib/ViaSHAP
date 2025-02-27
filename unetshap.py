import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fastshap.utils import ShapleySampler, MaskLayer2d
import os
from tqdm import tqdm
import gc
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


#  THE FOLLOWING UNET CODE IS NOT MINE, IT IS FROM https://github.com/iancovert/fastshap/blob/main/notebooks/unet.py and should be checked/credited
class MultiConv(nn.Module):
    '''(convolution => [BN] => ReLU) * n'''

    def __init__(self, in_channels, out_channels, mid_channels=None,
                 num_convs=2, batchnorm=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        if batchnorm:
            # Input conv.
            module_list = [
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)]

            # Middle convs.
            for _ in range(num_convs - 2):
                module_list = module_list + [
                    nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                              padding=1),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True)]

            # Output conv.
            module_list = module_list + [
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)]

        else:
            # Input conv.
            module_list = [
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)]

            # Middle convs.
            for _ in range(num_convs - 2):
                module_list = module_list + [
                    nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                              padding=1),
                    nn.ReLU(inplace=True)]

            # Output conv.
            module_list = module_list + [
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)]

        # Set up sequential.
        self.multi_conv = nn.Sequential(*module_list)

    def forward(self, x):
        return self.multi_conv(x)


class Down(nn.Module):
    '''
    Downscaling with maxpool then multiconv.
    Adapted from https://github.com/milesial/Pytorch-UNet
    '''

    def __init__(self, in_channels, out_channels, num_convs=2, batchnorm=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            MultiConv(in_channels, out_channels, num_convs=num_convs,
                      batchnorm=batchnorm)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    '''
    Upscaling then multiconv.
    Adapted from https://github.com/milesial/Pytorch-UNet
    '''

    def __init__(self, in_channels, out_channels, num_convs=2, bilinear=True,
                 batchnorm=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                                  align_corners=True)
            self.conv = MultiConv(in_channels, out_channels, in_channels // 2,
                                  num_convs, batchnorm)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                         kernel_size=2, stride=2)
            self.conv = MultiConv(in_channels, out_channels,
                                  num_convs=num_convs, batchnorm=batchnorm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self,
                 n_classes,
                 num_down,
                 num_up,
                 in_channels=3,
                 base_channels=64,
                 num_convs=2,
                 batchnorm=True,
                 bilinear=True):
        super().__init__()
        assert num_down >= num_up

        # Input conv.
        self.inc = MultiConv(in_channels, base_channels, num_convs=num_convs,
                             batchnorm=batchnorm)

        # Downsampling layers.
        down_layers = []
        channels = base_channels
        out_channels = 2 * channels
        for _ in range(num_down - 1):
            down_layers.append(
                Down(channels, out_channels, num_convs, batchnorm))
            channels = out_channels
            out_channels *= 2

        # Last down layer.
        factor = 2 if bilinear else 1
        down_layers.append(
            Down(channels, out_channels // factor, num_convs, batchnorm))
        self.down_layers = nn.ModuleList(down_layers)

        # Upsampling layers.
        up_layers = []
        channels *= 2
        out_channels = channels // 2
        for _ in range(num_up - 1):
            up_layers.append(
                Up(channels, out_channels // factor, num_convs, bilinear,
                   batchnorm))
            channels = out_channels
            out_channels = channels // 2

        # Last up layer.
        up_layers.append(
            Up(channels, out_channels, num_convs, bilinear, batchnorm))
        self.up_layers = nn.ModuleList(up_layers)

        # Output layer.
        self.outc = OutConv(out_channels, n_classes)

    def forward_unet(self, x):
        # Input conv.
        x = self.inc(x)

        # Apply downsampling layers.
        x_list = []
        for down in self.down_layers:
            x = down(x)
            x_list.append(x)

        # Apply upsampling layers.
        for i, up in enumerate(self.up_layers):
            residual_x = x_list[-(i + 2)]
            x = up(x, residual_x)

        # Output.
        logits = self.outc(x)
        return logits

    def predict_class(self, x):
        return x.sum(dim=-1).sum(dim=-1)

    def forward(self, x_in):
        x = self.forward_unet(x_in)

        x = self.predict_class(x)

        return x

    def predict(self, x_in):
        return self.forward(x_in)


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_roc = None
        self.best_epoch = None
        self.counter = 0

    def __call__(self, roc, epoch):
        if self.best_roc is None:
            self.best_roc = roc
            self.best_epoch = epoch
            return False

        if roc >= self.best_roc - self.min_delta:
            self.best_roc = roc
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False


def train_model(width, height, superpixel_size, train_dataloader,
                val_dataloader, data_name, num_classes,
                unet_model=None, optimizer_train=None, current_epoch=1,
                max_num_epochs=300, learning_rate=1e-03, num_samples=2,
                alpha=1, beta=16,
                patience=10):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    small_width = width // superpixel_size
    small_height = height // superpixel_size
    num_features = small_width * small_height

    sampler = ShapleySampler(num_features)

    if not os.path.exists(f'{data_name}'):
        os.makedirs(f'{data_name}')

    mse_loss = nn.MSELoss()
    softmax = nn.Softmax(dim=1)

    early_stopping = EarlyStopping(patience=patience)

    if unet_model is None:

        unet_model = UNet(num_classes, num_down=2, num_up=1, num_convs=3).to(device)

    if optimizer_train is None:
        optimizer_train = torch.optim.Adam(unet_model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_train, factor=0.5, patience=patience // 3, min_lr=1e-6,
        verbose=True)

    if superpixel_size == 1:
        upsample = nn.Identity()
    else:
        upsample = nn.Upsample(
            scale_factor=superpixel_size, mode='nearest')

    best_roc = -np.inf

    for epoch in range(current_epoch, max_num_epochs + 1):
        unet_model.train()

        train_loss = 0
        train_count = 0

        train_labels = []

        for i, data in enumerate(tqdm(train_dataloader)):
            inputs, labels = data
            labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer_train.zero_grad()

            outputs = unet_model(inputs)

            zeros = torch.zeros(1, inputs.shape[1], inputs.shape[2], inputs.shape[3], dtype=torch.float32,
                                device=device)

            null = unet_model(zeros)

            S = sampler.sample(inputs.shape[0] * num_samples,
                               paired_sampling=False)

            S_small = S.reshape(inputs.shape[0] * num_samples, small_height, small_width).unsqueeze(1).to(device)


            S = upsample(S_small)

            input_tiled = inputs.unsqueeze(1).repeat(
                1, num_samples, *[1 for _ in range(len(inputs.shape) - 1)]
            ).reshape(inputs.shape[0] * num_samples, *inputs.shape[1:]).to(device)

            sampled_inputs = input_tiled * S

            sampled_ouput = unet_model(sampled_inputs)


            scores = unet_model.forward_unet(inputs)


            scores_tiled = scores.unsqueeze(1).repeat(
                1, num_samples, *[1 for _ in range(len(inputs.shape) - 1)]
            ).reshape(inputs.shape[0] * num_samples, num_classes, small_height, small_width).to(device)


            sampled_scores = scores_tiled * S_small

            sampled_scores_output = null + unet_model.predict_class(sampled_scores)


            loss = alpha * mse_loss(outputs, labels) + beta * mse_loss(sampled_scores_output,
                                                                                 sampled_ouput)
            preds = torch.max(outputs, dim=-1)[1]
            train_count += torch.sum(preds == torch.max(labels, dim=-1)[1])


            train_loss += loss.item()  # * output.shape[0]

            loss.backward()
            optimizer_train.step()
            train_labels.extend(labels.tolist())

            del inputs, labels, data, outputs, S, input_tiled, sampled_inputs, sampled_ouput, scores, scores_tiled, sampled_scores, sampled_scores_output

            torch.cuda.empty_cache()
            gc.collect()

        unet_model.eval()

        val_count = 0
        val_loss = 0.0

        list_prediction = []
        val_labels = []
        list_prob_pred = []

        bce_loss = []
        int_loss = []
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_dataloader)):
                inputs, labels = data

                labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = softmax(unet_model(inputs))


                zeros = torch.zeros(1, inputs.shape[1], inputs.shape[2], inputs.shape[3], dtype=torch.float32,
                                    device=device)

                null = unet_model(zeros)

                S = sampler.sample(inputs.shape[0] * num_samples,
                                   paired_sampling=False)

                S_small = S.reshape(inputs.shape[0] * num_samples, small_height, small_width).unsqueeze(1).to(device)

                S = upsample(S_small)

                input_tiled = inputs.unsqueeze(1).repeat(
                    1, num_samples, *[1 for _ in range(len(inputs.shape) - 1)]
                ).reshape(inputs.shape[0] * num_samples, *inputs.shape[1:]).to(device)

                sampled_inputs = input_tiled * S

                sampled_ouput = unet_model(sampled_inputs)

                scores = unet_model.forward_unet(inputs)

                scores_tiled = scores.unsqueeze(1).repeat(
                    1, num_samples, *[1 for _ in range(len(inputs.shape) - 1)]
                ).reshape(inputs.shape[0] * num_samples, num_classes, small_height, small_width).to(device)

                sampled_scores = scores_tiled * S_small

                sampled_scores_output = null + unet_model.predict_class(sampled_scores)


                bce_loss.append(alpha * mse_loss(outputs, labels))
                int_loss.append(beta * mse_loss(sampled_scores_output, sampled_ouput))

                val_loss += (alpha * mse_loss(outputs, labels) + beta * mse_loss(sampled_scores_output,
                                                                                           sampled_ouput))
                preds = torch.max(outputs, dim=-1)[1]
                val_count += torch.sum(preds == torch.max(labels, dim=-1)[1])
                val_labels.extend(torch.max(labels, dim=-1)[1].tolist())
                list_prob_pred.extend(outputs.tolist())


                list_prediction.extend(preds.tolist())

                del inputs, labels, outputs, preds, S, input_tiled, sampled_inputs, sampled_ouput, scores, scores_tiled, sampled_scores, sampled_scores_output
                torch.cuda.empty_cache()
                gc.collect()

        roc = roc_auc_score(
            val_labels,
            list_prob_pred,
            multi_class="ovr",
            average="weighted",
        )


        prec = precision_score(val_labels, list_prediction, average='macro')
        recall = recall_score(val_labels, list_prediction, average='macro')
        f_score = f1_score(val_labels, list_prediction, average='macro')

        acc = "{:.3f}".format(val_count / len(val_labels))
        f_roc = "{:.6f}".format(roc)

        if best_roc <= roc:
            best_roc = roc
            f_roc = "{:.3f}".format(roc)
            torch.save(unet_model.state_dict(), f'{data_name}/{data_name}.model')
            torch.save(optimizer_train.state_dict(), f'{data_name}/{data_name}.optm')

        print(f'Classification loss: {sum(bce_loss) / len(bce_loss)}')
        print(f'interpretability loss: {sum(int_loss) / len(int_loss)}')

        print('Acc at dev is : {}'.format(acc))
        print('ROC is : {},  prec {},  recall {}, f-score {}'.format(f_roc, prec, recall, f_score))
        print('Acc at epoch : {} is : {}, loss : {}'.format(epoch,
                                                            train_count / len(train_labels), train_loss))

        scheduler.step(val_loss)

        if early_stopping(roc, epoch):
            print(f'Early stopping at epoch {epoch + 1}')
            print(f'Best validation result is at epoch: {early_stopping.best_epoch}')
            break
    unet_model.load_state_dict(torch.load(f'{data_name}/{data_name}.model'))
    return unet_model
