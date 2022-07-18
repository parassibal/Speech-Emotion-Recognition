import torch
from torch import nn
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from transformers import AutoModelForSequenceClassification
# from transformers import RobertaModel


class cnr(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(3, 3), stride=(1,1), padding=(1,1)):
        super(cnr, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class cnr1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(cnr1d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class vgg(nn.Module):
    def __init__(self):
        super(vgg, self).__init__()
        self.net = nn.Sequential(
            cnr(1, 32, kernel_size=(7, 7), stride=(3, 3), padding=(3, 3)),
            nn.MaxPool2d((2, 2)),
            cnr(32, 64, stride=(2, 2), padding=(0, 0)),
            cnr(64, 128, stride=(1, 1), padding=(0, 0)),
            cnr(128, 256, stride=(1, 1), padding=(0, 0)),
            nn.MaxPool2d((2, 2)),
            cnr(256, 512, stride=(1, 1), padding=(0, 0)),
            nn.MaxPool2d((2, 2)),
            cnr(512, 1024, stride=(1, 1), padding=(0, 0)),
            nn.Flatten(),
            # nn.LeakyReLU(negative_slope=1e-1),
            nn.Linear(1024, 8),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)


class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.con1 = cnr1d(256, 256, kernel_size=5, stride=1, padding=2)
        self.con2 = cnr1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.con3 = cnr1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.con4 = cnr1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.con5 = cnr1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.con6 = cnr1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.con7 = cnr1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.con8 = cnr1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.con9 = cnr1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.con10 = cnr1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.con11 = cnr1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.con12 = cnr1d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.maxpool1 = nn.MaxPool1d(5, 2, 1)
        self.con13 = cnr1d(1024, 1024, kernel_size=3, stride=2, padding=0)
        self.maxpool2 = nn.MaxPool1d(5, 2, 1)
        self.con14 = cnr1d(1024, 1024, kernel_size=3, stride=2, padding=0)
        self.maxpool3 = nn.MaxPool1d(5, 2, 1)
        self.con15 = cnr1d(1024, 1024, kernel_size=3, stride=2, padding=0)
        self.dp1 = nn.Dropout(0.2)
        self.dp2 = nn.Dropout(0.2)
        self.dp3 = nn.Dropout(0.2)
        self.linear_sequence = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 32),
            nn.LeakyReLU(negative_slope=1e-2),
            nn.Dropout(0.6),
            nn.Linear(32, 8),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.con1(x)
        x = self.con2(x)
        x3 = self.con3(x)
        x4 = self.con4(x3) + x3
        x5 = self.con5(x4) + x
        x6 = self.con6(x5) + x4
        x7 = self.con7(x6)
        x8 = self.con8(x7) + x6 + x5
        x9 = self.con9(x8)
        x10 = self.con10(x9) + x8
        x11 = self.con11(x10) + x8
        x = self.con12(x11)
        x = self.maxpool1(x)
        x = self.con13(x)
        x = self.maxpool2(x)
        x = self.con14(x)
        x = self.maxpool3(x)
        x = self.con15(x)
        x = self.linear_sequence(x)
        return x


class resnet_core(nn.Module):
    def __init__(self):
        super(resnet_core, self).__init__()
        self.con1 = cnr1d(256, 256, kernel_size=5, stride=1, padding=2)
        self.con2 = cnr1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.con3 = cnr1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.con4 = cnr1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.con5 = cnr1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.con6 = cnr1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.con7 = cnr1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.con8 = cnr1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.con9 = cnr1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.con10 = cnr1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.con11 = cnr1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.con12 = cnr1d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.maxpool1 = nn.MaxPool1d(5, 2, 1)
        self.con13 = cnr1d(1024, 1024, kernel_size=3, stride=2, padding=0)
        self.maxpool2 = nn.MaxPool1d(5, 2, 1)
        self.con14 = cnr1d(1024, 1024, kernel_size=3, stride=2, padding=0)
        self.maxpool3 = nn.MaxPool1d(5, 2, 1)
        self.con15 = cnr1d(1024, 1024, kernel_size=3, stride=2, padding=0)
        self.dp1 = nn.Dropout(0.2)
        self.dp2 = nn.Dropout(0.2)
        self.dp3 = nn.Dropout(0.2)
        self.linear_sequence = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 32),
        )

    def forward(self, x):
        x = self.con1(x)
        x = self.con2(x)
        x3 = self.con3(x)
        x4 = self.con4(x3) + x3
        x5 = self.con5(x4) + x
        x6 = self.con6(x5) + x4
        x7 = self.con7(x6)
        x8 = self.con8(x7) + x6 + x5
        x9 = self.con9(x8)
        x10 = self.con10(x9) + x8
        x11 = self.con11(x10) + x8
        x = self.con12(x11)
        x = self.maxpool1(x)
        x = self.con13(x)
        x = self.maxpool2(x)
        x = self.con14(x)
        x = self.maxpool3(x)
        x = self.con15(x)
        x = self.linear_sequence(x)
        return x


class vgg1d(nn.Module):
    def __init__(self):
        super(vgg1d, self).__init__()
        self.net = nn.Sequential(
            cnr1d(256, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2),
            cnr1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2),
            cnr1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2),
            cnr1d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.MaxPool1d(kernel_size=5, stride=2),
            cnr1d(1024, 1024, kernel_size=3, stride=1, padding=0),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(1024, 32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.6),
            nn.Linear(32, 5),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)


class cnn_lstm(nn.Module):
    def __init__(self):
        super(cnn_lstm, self).__init__()
        # self.conv1d = nn.Conv1d(256, 512, kernel_size=7, stride=3, padding=3)
        self.bilstm = nn.LSTM(256, 128, num_layers=4, dropout=0.2, bidirectional=True)
        self.linear = nn.Linear(256, 8)

    def forward(self, x):
        # x = self.conv1d(x)
        x, _ = self.bilstm(x)
        # x = nn.Flatten()(x)
        x = self.linear(x[:, -1, :])
        x = nn.Softmax(dim=1)(x)
        return x


class bert_model(nn.Module):
    def __init__(self, labels=5):
        super(bert_model, self).__init__()
        self.labels = labels
        self.bert = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=labels)

    def forward(self, ids, masks):
        return self.bert(ids, masks)[0]


class multi_modal(nn.Module):
    def __init__(self, labels=5):
        super(multi_modal, self).__init__()
        self.labels = labels
        self.bert = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=labels)
        # self.bert = RobertaModel.from_pretrained('roberta-base')
        self.vgg1dnet = nn.Sequential(
            cnr1d(256, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2),
            cnr1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2),
            cnr1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2),
            cnr1d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.MaxPool1d(kernel_size=5, stride=2),
            cnr1d(1024, 1024, kernel_size=3, stride=1, padding=0),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(1024, 32))
        self.resnet_core = resnet_core()
        self.linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32, labels),
            # nn.Softmax(dim=1)
            nn.Linear(labels, labels)
        )
        self.linear2 = nn.Linear(labels, labels)
        self.sm = nn.Softmax(dim=1)
        # self.bert_linear = nn.Linear(768, 32)

    def forward(self, mfcc, ids, masks):
        x1 = self.vgg1dnet(mfcc)
        x1 = self.linear(x1)
        # x1 = self.resnet_core(mfcc)
        x2 = self.bert(ids, masks)[0]
        x2 = self.linear2(x2)
        # x2 = self.bert_linear(x2)
        x2 = self.sm(x1 + x2)
        # x2 = self.linear(x2)
        return x2

