import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fastshap.utils import ShapleySampler, MaskLayer2d
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import gc


#  THE FOLLOWING RESNET PART OF THE CODE IS NOT MINE, IT IS FROM https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py and should be checked/credited
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, num_features, in_channels):
        super(ResNet, self).__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.in_planes = 64

        # Input conv.
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual blocks.
        channels = 64
        stride = 1
        blocks = []
        for num in num_blocks[:-1]:
            blocks.append(self._make_layer(block, channels, num, stride=stride))
            channels *= 2
            stride = 1

        num = num_blocks[-1]
        blocks.append(self._make_layer(block, channels, num, stride=2))

        self.layers = nn.ModuleList(blocks)

        # Output layer.
        self.num_classes = num_classes
        self.num_features = num_features
        if num_classes is not None:

            self.outc = OutConv(512*block.expansion, num_classes)



    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_resnet(self, x):
        # Input conv.
        out = F.relu(self.bn1(self.conv1(x)))

        # Residual blocks.
        for layer in self.layers:
            out = layer(out)

        # Output layer.
        if self.num_classes is not None:
            #out = F.avg_pool2d(out, 4)
            logits = self.outc(out)

        return logits

    def predict_class(self, x):
        return x.sum(dim=-1).sum(dim=-1)

    def forward(self, x_in):
        x = self.forward_resnet(x_in)

        x = self.predict_class(x)

        return x

    def predict(self, x_in):
        return self.forward(x_in)


def ResNet18(num_classes, num_features, in_channels=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, num_features, in_channels)


def ResNet34(num_classes, num_features, in_channels=3):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, num_features, in_channels)


def ResNet50(num_classes, num_features, in_channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, num_features, in_channels)


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
                resnet_model=None, optimizer_train=None, current_epoch=1,
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

    if resnet_model is None:

        resnet_model = ResNet18(num_classes, num_features).to(device)

    if optimizer_train is None:
        optimizer_train = torch.optim.Adam(resnet_model.parameters(), lr=learning_rate)

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
        resnet_model.train()

        train_loss = 0
        train_count = 0

        train_labels = []

        for i, data in enumerate(tqdm(train_dataloader)):
            inputs, labels = data
            labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer_train.zero_grad()

            outputs = resnet_model(inputs)

            zeros = torch.zeros(1, inputs.shape[1], inputs.shape[2], inputs.shape[3], dtype=torch.float32,
                                device=device)

            null = resnet_model(zeros)

            S = sampler.sample(inputs.shape[0] * num_samples,
                               paired_sampling=False)

            S_small = S.reshape(inputs.shape[0] * num_samples, small_height, small_width).unsqueeze(1).to(device)


            S = upsample(S_small)

            input_tiled = inputs.unsqueeze(1).repeat(
                1, num_samples, *[1 for _ in range(len(inputs.shape) - 1)]
            ).reshape(inputs.shape[0] * num_samples, *inputs.shape[1:]).to(device)

            sampled_inputs = input_tiled * S

            sampled_ouput = resnet_model(sampled_inputs)


            scores = resnet_model.forward_resnet(inputs)

            scores_tiled = scores.unsqueeze(1).repeat(
                1, num_samples, *[1 for _ in range(len(inputs.shape) - 1)]
            ).reshape(inputs.shape[0] * num_samples, num_classes, small_height, small_width).to(device)


            sampled_scores = scores_tiled * S_small

            sampled_scores_output = null + resnet_model.predict_class(sampled_scores)


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

        resnet_model.eval()

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

                outputs = softmax(resnet_model(inputs))


                zeros = torch.zeros(1, inputs.shape[1], inputs.shape[2], inputs.shape[3], dtype=torch.float32,
                                    device=device)

                null = resnet_model(zeros)

                S = sampler.sample(inputs.shape[0] * num_samples,
                                   paired_sampling=False)

                S_small = S.reshape(inputs.shape[0] * num_samples, small_height, small_width).unsqueeze(1).to(device)

                S = upsample(S_small)

                input_tiled = inputs.unsqueeze(1).repeat(
                    1, num_samples, *[1 for _ in range(len(inputs.shape) - 1)]
                ).reshape(inputs.shape[0] * num_samples, *inputs.shape[1:]).to(device)

                sampled_inputs = input_tiled * S

                sampled_ouput = resnet_model(sampled_inputs)

                scores = resnet_model.forward_resnet(inputs)

                scores_tiled = scores.unsqueeze(1).repeat(
                    1, num_samples, *[1 for _ in range(len(inputs.shape) - 1)]
                ).reshape(inputs.shape[0] * num_samples, num_classes, small_height, small_width).to(device)

                sampled_scores = scores_tiled * S_small

                sampled_scores_output = null + resnet_model.predict_class(sampled_scores)


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
            torch.save(resnet_model.state_dict(), f'{data_name}/{data_name}.model')
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
    resnet_model.load_state_dict(torch.load(f'{data_name}/{data_name}.model'))
    return resnet_model
