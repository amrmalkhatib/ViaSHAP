import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
from fastshap.utils import ShapleySampler
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


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
        # self.fc_numerical = nn.Linear(1, emb_dim)

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
                # numerical_x = self.fc_numerical(numerical_x)

                # Concatenate numerical and categorical embeddings
                embedded_x = torch.cat([numerical_x, nominal_x], dim=1)
            else:
                nominal_x = x
                # List to store embeddings for each categorical column
                embedded = [self.embedding_dict[col](nominal_x[:, i].long()) for i, col in
                            enumerate(self.categorical)]
                embedded_x = torch.cat(embedded, dim=1)
        else:
            embedded_x = x  # self.fc_numerical(x)

        return embedded_x



class MLPSHAP(nn.Module):
    def __init__(self, num_features, num_classes, index_to_name, dataframe, categorical, numerical, link_func=False, add_bias=True):
        super(MLPSHAP, self).__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding = Embed(dataframe, categorical, numerical, 32)

        self.fc1 = nn.Linear(len(categorical) * 32 + len(numerical), 64)

        self.bn1 = nn.BatchNorm1d(64)

        self.fc2 = nn.Linear(64, 128)

        self.bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 64)

        self.bn3 = nn.BatchNorm1d(64)

        self.fc4 = nn.Linear(64, num_features * num_classes)

        self.num_classes = num_classes
        self.num_features = num_features

        self.link_func = link_func
        self.add_bias = add_bias

        if add_bias:
            self.delta = torch.nn.init.uniform_(Parameter(torch.FloatTensor(num_classes)), -0.1, 0.1)

        self.O = torch.ones(self.num_features, 1, requires_grad=False).to(device)

        self.relu = nn.ReLU()

        if link_func:
            self.sigmoid = nn.Sigmoid()
            self.softmax = nn.Softmax(dim=1)

        self.index_to_name = index_to_name

    def mlp_forward(self, x_in):

        x = self.embedding(x_in)

        x = self.relu(self.bn1(self.fc1(x)))

        x = self.relu(self.bn2(self.fc2(x)))

        x = self.relu(self.bn3(self.fc3(x)))

        x = self.fc4(x)

        return x

    def predict_class(self, x):

        if self.num_classes > 1:
            x = x.view(x.size(0), self.num_classes, self.num_features) @ self.O

            if self.add_bias:
                x = torch.squeeze(x, 2) + self.delta
            else:
                x = torch.squeeze(x, 2)
            if self.link_func:
                x = self.softmax(x)
        else:
            if self.add_bias:
                x = torch.mm(x.view(x.size(0), x.size(1)), self.O) + self.delta
            else:
                x = torch.mm(x.view(x.size(0), x.size(1)), self.O)
            if self.link_func:
                x = self.sigmoid(x)
        return x


    def forward(self, x_in):
        x = self.mlp_forward(x_in)

        x = self.predict_class(x)

        return x

    def predict(self, x_in):
        return self.forward(x_in)

    def get_local_importance(self, x_in):

        if self.num_classes > 1:
            y = self.forward(x_in)
            y = np.argmax(y[0].cpu().detach().numpy())

            x = self.mlp_forward(x_in)
            x = x.view(x.size(0), self.num_classes, self.num_features)
            x = x[:, y].reshape(-1)
        else:
            x = self.mlp_forward(x_in)
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

        # if self.num_classes > 2:
        #    center = 0
        # else:
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
                mlp_model=None, optimizer_train=None, current_epoch=1,
                max_num_epochs=300, learning_rate=1e-03, num_samples=16,
                alpha=0.5, beta=0.5,
                link_func=False, add_bias=True,
                efficiency_constraint=True, patience=20):

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

    if mlp_model is None:
        mlp_model = MLPSHAP(num_features, num_classes, index_to_name, dataframe, categorical,
                            numerical, link_func, add_bias).to(device)
    if optimizer_train is None:
        optimizer_train = torch.optim.Adam(mlp_model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_train, factor=0.5, patience=patience // 3, min_lr=1e-6,
        verbose=True)

    best_val_loss = np.inf

    for epoch in range(current_epoch, max_num_epochs + 1):
        mlp_model.train()

        train_loss = 0
        train_count = 0

        train_labels = []

        for i, data in enumerate(tqdm(train_dataloader)):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer_train.zero_grad()

            outputs = mlp_model(inputs)

            if efficiency_constraint:
                zeros = torch.zeros(2, num_features, dtype=torch.float32,
                                    device=device)

                null = mlp_model(zeros)[0].unsqueeze(0)

            S = sampler.sample(inputs.shape[0] * num_samples,
                               paired_sampling=True)
            S = S.reshape((inputs.shape[0] * num_samples, num_features)).to(device)

            input_tiled = inputs.unsqueeze(1).repeat(
                1, num_samples, *[1 for _ in range(len(inputs.shape) - 1)]
            ).reshape(inputs.shape[0] * num_samples, *inputs.shape[1:]).to(device)

            sampled_inputs = input_tiled * S

            sampled_ouput = mlp_model(sampled_inputs)

            scores = mlp_model.mlp_forward(inputs)

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

            if efficiency_constraint:
                sampled_scores_output = null + mlp_model.predict_class(sampled_scores)
            else:
                sampled_scores_output = mlp_model.predict_class(sampled_scores)

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

        mlp_model.eval()

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

            if efficiency_constraint:
                zeros = torch.zeros(1, num_features, dtype=torch.float32,
                                    device=device)

                null = mlp_model(zeros)

            S = sampler.sample(inputs.shape[0] * num_samples,
                               paired_sampling=True)
            S = S.reshape((inputs.shape[0] * num_samples, num_features)).to(device)

            input_tiled = inputs.unsqueeze(1).repeat(
                1, num_samples, *[1 for _ in range(len(inputs.shape) - 1)]
            ).reshape(inputs.shape[0] * num_samples, *inputs.shape[1:]).to(device)

            sampled_inputs = input_tiled * S

            sampled_ouput = mlp_model(sampled_inputs)

            scores = mlp_model.mlp_forward(inputs)

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

            if efficiency_constraint:
                sampled_scores_output = null + mlp_model.predict_class(sampled_scores)
            else:
                sampled_scores_output = mlp_model.predict_class(sampled_scores)

            outputs = mlp_model(inputs)

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
            torch.save(mlp_model.state_dict(), f'{data_name}/{data_name}_mlp.model')
            torch.save(optimizer_train.state_dict(), f'{data_name}/{data_name}_mlp.optm')

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
    mlp_model.load_state_dict(torch.load(f'{data_name}/{data_name}_mlp.model'))
    return mlp_model
