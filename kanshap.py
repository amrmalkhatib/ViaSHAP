import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from fastshap.utils import ShapleySampler
from ekan import KAN as kan
import math
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


class Block(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Block, self).__init__()

        self.fc_1a = nn.Linear(input_dim, output_dim)
        self.fc_1b = nn.Linear(output_dim, output_dim)

        self.fc_2a = nn.Linear(input_dim, output_dim)
        self.fc_2b = nn.Linear(output_dim, output_dim)
        self.concat = nn.Linear(output_dim + output_dim, output_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.cat((self.fc_1b(self.fc_1a(x)),
                       self.fc_2b(self.fc_2a(x))), 2)
        x = self.relu(self.concat(x))
        return x


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.25, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, h, adj):
        # h.shape: (batch_size, N, in_features)
        # adj.shape: (batch_size, N, N)

        # Linear transformation
        Wh = torch.matmul(h, self.W)  # h.shape: (batch_size, N, in_features), Wh.shape: (batch_size, N, out_features)

        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)  # Softmax along the last dimension
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, Wh)  # h_prime.shape: (batch_size, N, out_features)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.transpose(1, 2)  # Transpose to match dimensions
        return self.leakyrelu(e)


class AttBlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4):
        super(AttBlock, self).__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads

        self.query1 = nn.Linear(input_dim, output_dim)
        self.key1 = nn.Linear(input_dim, output_dim)
        self.value1 = nn.Linear(input_dim, output_dim)

        self.multihead_attn1 = nn.MultiheadAttention(output_dim, self.num_heads)

    def forward(self, x):
        query = self.query1(x)
        key = self.key1(x)
        value = self.value1(x)

        weighted = self.multihead_attn1(query, key, value)

        return weighted[0]


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.best_epoch = None
        self.counter = 0

    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
            return False

        if val_loss <= self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False


class Embed(nn.Module):
    def __init__(self, dataframe, categorical, numerical, emb_dim):
        super(Embed, self).__init__()

        self.categorical = categorical
        self.numerical = numerical
        self.emb_dim = emb_dim

        # Create a dictionary to store embeddings for each categorical column
        self.embedding_dict = nn.ModuleDict()

        for col in self.categorical:
            num_emb = dataframe[col].nunique()
            self.embedding_dict[col] = nn.Embedding(num_emb + 1, emb_dim)

        # Linear layer for numerical features
        #self.fc_numerical = nn.Linear(1, emb_dim)

    def forward(self, x):
        if len(self.categorical) > 0:
            if len(self.numerical) > 0:
                numerical_x = x[:, :len(self.numerical)]
                nominal_x = x[:, len(self.numerical):]

                # List to store embeddings for each categorical column
                embedded = [self.embedding_dict[col](nominal_x[:, i].long()) for i, col in
                            enumerate(self.categorical)]
                nominal_x = torch.cat(embedded, dim=1)

                # Apply linear transformation to numerical features
                #numerical_x = self.fc_numerical(numerical_x)

                # Concatenate numerical and categorical embeddings
                embedded_x = torch.cat([numerical_x, nominal_x], dim=1)
            else:
                nominal_x = x
                # List to store embeddings for each categorical column
                embedded = [self.embedding_dict[col](nominal_x[:, i].long()) for i, col in
                            enumerate(self.categorical)]
                embedded_x = torch.cat(embedded, dim=1)
        else:
            embedded_x = x#self.fc_numerical(x)

        return embedded_x


class FNN(nn.Module):
    def __init__(self, num_features):
        super(FNN, self).__init__()

        self.fc_1 = nn.Linear(64 + 256 + 256, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.bn1 = nn.BatchNorm1d(num_features)

        self.fc_3 = nn.Linear(64, 32)
        self.fc_4 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(num_features)

        self.fc_5 = nn.Linear(16, 8)
        self.fc_6 = nn.Linear(8, 4)
        self.bn3 = nn.BatchNorm1d(num_features)

        self.fc_7 = nn.Linear(4, 2)
        self.fc_8 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.bn1(self.fc_2(self.fc_1(x)))

        x = self.bn2(self.fc_4(self.fc_3(x)))

        x = self.bn3(self.fc_6(self.fc_5(x)))

        x = self.fc_8(self.fc_7(x))

        return x


def normalize_adj_matrix(adj):
    D = torch.sum(adj, 0)
    D_hat = torch.diag(((D) ** (-0.5)))
    adj_normalized = torch.mm(torch.mm(D_hat, adj), D_hat)
    return adj_normalized


class KANSHAP(nn.Module):
    def __init__(self, num_features, num_classes, index_to_name, dataframe, categorical, numerical):
        super(KANSHAP, self).__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding = Embed(dataframe, categorical, numerical, 32)

        self.f1 = kan([len(categorical)*32 + len(numerical), 64, 128, 64, num_features * num_classes], grid_size=5, spline_order=3)

        #self.f2 = kan([128, 64, num_features], grid_size=5, spline_order=3)

        self.num_classes = num_classes
        self.num_features = num_features


        #if num_classes > 2:
        #    self.O = torch.ones(self.num_features, self.num_classes, requires_grad=False).to(device)
        #else:
        self.O = torch.ones(self.num_features, 1, requires_grad=False).to(device)


        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.index_to_name = index_to_name


    def gnn_forward(self, x_in):

        x = self.embedding(x_in)

        x = self.f1(x)

        #x = self.f2(x)

        return x

    def predict_class(self, x):

        if self.num_classes > 1:
            x = x.view(x.size(0), self.num_classes, self.num_features) @ self.O
            x = torch.squeeze(x, 2)
            x = self.softmax(x)
        else:
            x = torch.mm(x.view(x.size(0), x.size(1)), self.O)
            x = self.sigmoid(x)
        return x

    def forward(self, x_in):
        x = self.gnn_forward(x_in)

        x = self.predict_class(x)

        return x

    def predict(self, x_in):
        return self.forward(x_in)

    def get_local_importance(self, x_in):

        if self.num_classes > 1:
            y = self.forward(x_in)
            y = np.argmax(y[0].cpu().detach().numpy())

            x = self.gnn_forward(x_in)
            x = x.view(x.size(0), self.num_classes, self.num_features)
            x = x[:, y].reshape(-1)
        else:
            x = self.gnn_forward(x_in)
            x = x.view(x.size(0), x.size(1)).reshape(-1)

        return x.cpu().data.numpy()


    def plot_bars(self, normalized_instance, instance, num_f):
        import matplotlib.pyplot as plt

        local_importance = self.get_local_importance(normalized_instance)

        original_values = instance.to_dict()

        names = []
        values = []
        for i, v in enumerate(local_importance):
            name = self.index_to_name[i]
            names.append(name)
            values.append(v)

        feature_local_importance = {}
        for i, v in enumerate(values):
            feature_local_importance[self.index_to_name[i]] = v

        feature_names = [f'{name} = {original_values[name]}' for name, val in sorted(feature_local_importance.items(),
                                                                                     key=lambda item: abs(item[1]))]
        feature_values = [val for name, val in sorted(feature_local_importance.items(),
                                                      key=lambda item: abs(item[1]))]

        plt.style.use('ggplot')
        plt.rcParams.update({'font.size': 12, 'font.weight': 'bold'})

        #if self.num_classes > 2:
        #    center = 0
        #else:
         #   center = self.beta.item()
        plt.barh(feature_names[-num_f:], feature_values[-num_f:], left=0,
                 color=np.where(np.array(feature_values[-num_f:]) < 0, 'dodgerblue', '#f5054f'))

        for index, v in enumerate(feature_values[-num_f:]):
            if v > 0:
                plt.text(v + 0, index, "+{:.2f}".format(v), ha='center')
            else:
                plt.text(v + 0, index, "{:.2f}".format(v), ha='left')

        plt.xlabel('Importance')
        plt.rcParams["figure.figsize"] = (8, 8)

        plt.show()


def train_model(index_to_name, train_dataloader,
                val_dataloader, data_name, num_classes,
                dataframe, categorical, numerical,
                gnn_model=None, optimizer_train=None, current_epoch=1,
                max_num_epochs=300, learning_rate=1e-03, num_samples=16,
                alpha=0.5, beta=0.5,
                patience=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_features = len(index_to_name)

    sampler = ShapleySampler(num_features)


    if not os.path.exists(f'{data_name}'):
        os.makedirs(f'{data_name}')

    mse_loss = nn.MSELoss()
    if num_classes > 2:
        loss_function = nn.MSELoss()
    else:
        loss_function = nn.BCELoss()

    early_stopping = EarlyStopping(patience=patience)

    if gnn_model is None:
        gnn_model = KANSHAP(num_features, num_classes, index_to_name, dataframe, categorical,
                            numerical).to(device)
    if optimizer_train is None:
        optimizer_train = torch.optim.Adam(gnn_model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_train, factor=0.5, patience=patience // 3, min_lr=1e-6,
        verbose=True)

    best_val_loss = np.inf

    for epoch in range(current_epoch, max_num_epochs + 1):
        gnn_model.train()

        train_loss = 0
        train_count = 0

        train_labels = []


        for i, data in enumerate(tqdm(train_dataloader)):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer_train.zero_grad()

            outputs = gnn_model(inputs)

            zeros = torch.zeros(1, num_features, dtype=torch.float32,
                                device=device)

            null = gnn_model(zeros)

            S = sampler.sample(inputs.shape[0] * num_samples,
                               paired_sampling=True)
            S = S.reshape((inputs.shape[0] * num_samples, num_features)).to(device)

            input_tiled = inputs.unsqueeze(1).repeat(
                1, num_samples, *[1 for _ in range(len(inputs.shape) - 1)]
            ).reshape(inputs.shape[0] * num_samples, *inputs.shape[1:]).to(device)

            sampled_inputs = input_tiled * S

            sampled_ouput = gnn_model(sampled_inputs)

            scores = gnn_model.gnn_forward(inputs)

            if num_classes > 1:
                scores_tiled = scores.unsqueeze(1).repeat(
                    1, num_samples, *[1 for _ in range(len(inputs.shape) - 1)]
                ).reshape(inputs.shape[0] * num_samples, num_classes, *inputs.shape[1:]).to(device)

                S = S.unsqueeze(1)
            else:
                scores_tiled = scores.unsqueeze(1).repeat(
                    1, num_samples, *[1 for _ in range(len(inputs.shape) - 1)]
                ).reshape(inputs.shape[0] * num_samples, *inputs.shape[1:]).to(device)

            sampled_scores = scores_tiled * S
            sampled_scores_output = null + gnn_model.predict_class(sampled_scores)

            if num_classes > 1:

                loss = alpha * loss_function(outputs, labels) + beta * loss_function(sampled_scores_output,
                                                                                     sampled_ouput)
                preds = torch.max(outputs, dim=-1)[1]
                train_count += torch.sum(preds == torch.max(labels, dim=-1)[1])
            else:

                loss = alpha * loss_function(outputs.reshape(-1), labels.float()) + \
                       beta * mse_loss(sampled_scores_output.reshape(-1), sampled_ouput.reshape(-1))
                preds = (outputs.reshape(-1) > 0.5) * 1
                train_count += torch.sum(preds == labels.data)

            train_loss += loss.item()  # * output.shape[0]

            loss.backward()
            optimizer_train.step()
            train_labels.extend(labels.tolist())

            del inputs, labels, data, outputs, S, input_tiled, sampled_inputs, sampled_ouput, scores, scores_tiled, sampled_scores, sampled_scores_output

            torch.cuda.empty_cache()

        gnn_model.eval()

        val_count = 0
        val_loss = 0.0

        list_prediction = []
        val_labels = []
        list_prob_pred = []

        bce_loss = []
        int_loss = []

        for i, data in enumerate(tqdm(val_dataloader)):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            zeros = torch.zeros(1, num_features, dtype=torch.float32,
                                device=device)

            null = gnn_model(zeros)

            S = sampler.sample(inputs.shape[0] * num_samples,
                               paired_sampling=True)
            S = S.reshape((inputs.shape[0] * num_samples, num_features)).to(device)

            input_tiled = inputs.unsqueeze(1).repeat(
                1, num_samples, *[1 for _ in range(len(inputs.shape) - 1)]
            ).reshape(inputs.shape[0] * num_samples, *inputs.shape[1:]).to(device)

            sampled_inputs = input_tiled * S

            sampled_ouput = gnn_model(sampled_inputs)

            scores = gnn_model.gnn_forward(inputs)

            if num_classes > 1:
                scores_tiled = scores.unsqueeze(1).repeat(
                    1, num_samples, *[1 for _ in range(len(inputs.shape) - 1)]
                ).reshape(inputs.shape[0] * num_samples, num_classes, *inputs.shape[1:]).to(device)

                S = S.unsqueeze(1)
            else:
                scores_tiled = scores.unsqueeze(1).repeat(
                    1, num_samples, *[1 for _ in range(len(inputs.shape) - 1)]
                ).reshape(inputs.shape[0] * num_samples, *inputs.shape[1:]).to(device)

            sampled_scores = scores_tiled * S
            sampled_scores_output = null + gnn_model.predict_class(sampled_scores)

            outputs = gnn_model(inputs)

            if num_classes > 1:
                bce_loss.append(alpha * loss_function(outputs, labels))
                int_loss.append(beta * loss_function(sampled_scores_output, sampled_ouput))

                val_loss += (alpha * loss_function(outputs, labels) + beta * loss_function(sampled_scores_output,
                                                                                sampled_ouput))
                preds = torch.max(outputs, dim=-1)[1]
                val_count += torch.sum(preds == torch.max(labels, dim=-1)[1])
                val_labels.extend(torch.max(labels, dim=-1)[1].tolist())
                list_prob_pred.extend(outputs.tolist())
            else:
                bce_loss.append(alpha * loss_function(outputs.reshape(-1), labels.float()))
                int_loss.append(beta * mse_loss(sampled_scores_output.reshape(-1), sampled_ouput.reshape(-1)))

                val_loss += (alpha * loss_function(outputs.reshape(-1), labels.float()) + \
                       beta * mse_loss(sampled_scores_output.reshape(-1), sampled_ouput.reshape(-1)))
                preds = (outputs.reshape(-1) > 0.5) * 1
                val_count += torch.sum(preds == labels.data)
                val_labels.extend(labels.tolist())

            list_prediction.extend(preds.tolist())

            del inputs, labels, outputs, preds, S, input_tiled, sampled_inputs, sampled_ouput, scores, scores_tiled, sampled_scores, sampled_scores_output
            torch.cuda.empty_cache()

        if num_classes > 2:
            roc = roc_auc_score(
                val_labels,
                list_prob_pred,
                multi_class="ovr",
                average="weighted",
            )
        else:
            roc = roc_auc_score(val_labels, list_prediction)

        prec = precision_score(val_labels, list_prediction, average='macro')
        recall = recall_score(val_labels, list_prediction, average='macro')
        f_score = f1_score(val_labels, list_prediction, average='macro')

        acc = "{:.3f}".format(val_count / len(val_labels))
        f_roc = "{:.6f}".format(roc)

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            f_roc = "{:.3f}".format(roc)
            torch.save(gnn_model.state_dict(), f'{data_name}/{data_name}.model')
            torch.save(optimizer_train.state_dict(), f'{data_name}/{data_name}.optm')

        print(f'Classification loss: {sum(bce_loss) / len(bce_loss)}')
        print(f'interpretability loss: {sum(int_loss) / len(int_loss)}')

        print('Acc at dev is : {}'.format(acc))
        print('ROC is : {},  prec {},  recall {}, f-score {}'.format(f_roc, prec, recall, f_score))
        print('Acc at epoch : {} is : {}, loss : {}'.format(epoch,
                                                            train_count / len(train_labels), train_loss))

        scheduler.step(val_loss)

        if early_stopping(val_loss, epoch):
            print(f'Early stopping at epoch {epoch + 1}')
            print(f'Best validation result is at epoch: {early_stopping.best_epoch}')
            break

    #torch.save(gnn_model.state_dict(), f'{data_name}/{data_name}-epoch[{epoch}].model')
    #torch.save(optimizer_train.state_dict(), f'{data_name}/{data_name}-epoch[{epoch}].optm')
    return gnn_model
