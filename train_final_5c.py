import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import time
import copy
from tqdm import tqdm
from dataloader_final_5c import *
from model_final_5c import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_multi(dataloader_train, dataloader_val, model, criterion, optimizer, num_epochs, dataset_size):
    model.to(device)
    # since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    dataloaders = {'train': dataloader_train, 'val': dataloader_val}
    loss_re = {'train': [], 'val': []}
    acc_re = {'train': [], 'val': []}
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()

            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, ids, masks, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                ids = ids.to(device)
                masks = masks.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, ids, masks)
                    _, pred = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(pred==labels.data)

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects / dataset_size[phase]
            loss_re[phase].append(epoch_loss)
            acc_re[phase].append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        torch.save(best_model_wts, 'best_model_multi.pth')
        scheduler.step()
    return loss_re, acc_re


def train_cnn(dataloader_train, dataloader_val, model, criterion, optimizer, num_epochs, dataset_size):
    model.to(device)
    # since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    dataloaders = {'train': dataloader_train, 'val': dataloader_val}
    loss_re = {'train': [], 'val': []}
    acc_re = {'train': [], 'val': []}
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()

            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, ids, masks, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                ids = ids.to(device)
                masks = masks.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, pred = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(pred==labels.data)

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects / dataset_size[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        torch.save(best_model_wts, 'best_model_cnn.pth')
        scheduler.step()
    return loss_re, acc_re


def train_bert(dataloader_train, dataloader_val, model, criterion, optimizer, num_epochs, dataset_size):
    model.to(device)
    # since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    dataloaders = {'train': dataloader_train, 'val': dataloader_val}
    loss_re = {'train': [], 'val': []}
    acc_re = {'train': [], 'val': []}
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()

            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, ids, masks, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                ids = ids.to(device)
                masks = masks.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(ids, masks)
                    _, pred = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(pred==labels.data)

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects / dataset_size[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        torch.save(best_model_wts, 'best_model_bert.pth')
        scheduler.step()
    return loss_re, acc_re


def test_bert(dataloader_test, model, dataset_size, criterion):
    model.load_state_dict(torch.load('best_model_bert.pth'))
    model.to(device)
    model.eval()
    running_loss = 0.0
    running_corrects = 0.0
    res = []
    for inputs, ids, masks, labels in tqdm(dataloader_test):
        inputs = inputs.to(device)
        labels = labels.to(device)
        ids = ids.to(device)
        masks = masks.to(device)

        outputs = model(ids, masks)
        _, pred = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(pred == labels.data)
        for i in outputs.cpu().detach().numpy():
            res.append(i)
    epoch_loss = running_loss / dataset_size['test']
    epoch_acc = running_corrects / dataset_size['test']
    print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    number = []
    import csv
    with open(f'./test.csv') as f:
        freader = csv.reader(f)
        t = 0
        for i in freader:
            if t >= 1:
                number.append(i[0])
            t += 1
    assert(len(res) == len(number))
    logits = {}
    for i in range(len(res)):
        logits[number[i]] = list(res[i].astype(np.float64))
    import json
    with open('bert_res.json', 'w') as fp:
        json.dump(logits, fp)


def test_cnn(dataloader_test, model, dataset_size, criterion):
    model.load_state_dict(torch.load('best_model_cnn.pth'))
    model.to(device)
    model.eval()
    running_loss = 0.0
    running_corrects = 0.0
    res = []
    for inputs, ids, masks, labels in tqdm(dataloader_test):
        inputs = inputs.to(device)
        labels = labels.to(device)
        ids = ids.to(device)
        masks = masks.to(device)

        outputs = model(inputs)
        _, pred = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(pred == labels.data)
        for i in outputs.cpu().detach().numpy():
            res.append(i)
    epoch_loss = running_loss / dataset_size['test']
    epoch_acc = running_corrects / dataset_size['test']
    print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    number = []
    import csv
    with open(f'./test.csv') as f:
        freader = csv.reader(f)
        t = 0
        for i in freader:
            if t >= 1:
                number.append(i[0])
            t += 1
    assert(len(res) == len(number))
    logits = {}
    for i in range(len(res)):
        logits[number[i]] = list(res[i].astype(np.float64))
    import json
    with open('cnn_res.json', 'w') as fp:
        json.dump(logits, fp)


def test_multi(dataloader_test, model, dataset_size, criterion):
    model.load_state_dict(torch.load('best_model_multi.pth'))
    model.to(device)
    model.eval()
    running_loss = 0.0
    running_corrects = 0.0
    res = []
    for inputs, ids, masks, labels in tqdm(dataloader_test):
        inputs = inputs.to(device)
        labels = labels.to(device)
        ids = ids.to(device)
        masks = masks.to(device)

        outputs = model(inputs, ids, masks)
        _, pred = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(pred == labels.data)
        for i in outputs.cpu().detach().numpy():
            res.append(i)
    epoch_loss = running_loss / dataset_size['test']
    epoch_acc = running_corrects / dataset_size['test']
    print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    number = []
    import csv
    with open(f'./test.csv') as f:
        freader = csv.reader(f)
        t = 0
        for i in freader:
            if t >= 1:
                number.append(i[0])
            t += 1
    assert(len(res) == len(number))
    logits = {}
    for i in range(len(res)):
        logits[number[i]] = list(res[i].astype(np.float64))
    import json
    with open('multi_res.json', 'w') as fp:
        json.dump(logits, fp)


if __name__ == '__main__':
    model = multi_modal()
    dataloaders = dataset_create_5c()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1.5e-4, weight_decay=1e-3)
    dataset_size = {'train': 5904, 'val': 1476}
    train_multi(dataloaders, model, criterion, optimizer, 15, dataset_size)