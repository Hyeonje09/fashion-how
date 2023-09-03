from dataset import ETRIDataset_emo
from networks import *

import pandas as pd
import os
import argparse
import time

import torch
import torch.utils.data
import torch.utils.data.distributed


parser = argparse.ArgumentParser()
parser.add_argument("--version", type=str, default='Baseline_ResNet_emo')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', default=0.0001, type=float, metavar='N',
                    help='learning rate')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

a, _ = parser.parse_known_args()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    if os.path.exists('models') is False:
        os.makedirs('models')

    save_path = 'models/' + a.version
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    net = Baseline_ResNet_emo().to(DEVICE)

    df = pd.read_csv('./Dataset/info_etri20_emotion_train.csv')
    train_dataset = ETRIDataset_emo(df, base_path='./Dataset/Train/')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=a.batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(net.parameters(), lr=a.lr)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    total_step = len(train_dataloader)
    step = 0
    t0 = time.time()

    for epoch in range(a.epochs):
        net.train()

        for i, sample in enumerate(train_dataloader):
            optimizer.zero_grad()
            step += 1
            for key in sample:
                sample[key] = sample[key].to(DEVICE)

            out_daily, out_gender, out_embel = net(sample)

            loss_daily = criterion(out_daily, sample['daily_label'])
            loss_gender = criterion(out_gender, sample['gender_label'])
            loss_embel = criterion(out_embel, sample['embel_label'])
            loss = loss_daily + loss_gender + loss_embel

            loss.backward() 
            optimizer.step()

            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, '
                      'Loss_daily: {:.4f}, Loss_gender: {:.4f}, Loss_embel: {:.4f}, Time : {:2.3f}'
                      .format(epoch + 1, a.epochs, i + 1, total_step, loss.item(), 
                              loss_daily.item(), loss_gender.item(), loss_embel.item(), time.time() - t0))

                t0 = time.time()

        if ((epoch + 1) % 10 == 0):
            a.lr *= 0.9
            optimizer = torch.optim.Adam(net.parameters(), lr=a.lr)
            print("learning rate is decayed")


        if ((epoch + 1) % 20 == 0):
            print('Saving Model....')
            torch.save(net.state_dict(), save_path + '/model_' + str(epoch + 1) + '.pkl')
            print('OK.')



if __name__ == '__main__':
    main()

