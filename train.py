import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import random_split
from torch import nn

import argparse
import random
import numpy as np
from tqdm import tqdm
import sys
from matplotlib import pyplot as plt
from setproctitle import setproctitle
import os

from models.vit import VisionTransformer

# シードの設定を行う関数
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# データセットのロードを行う関数
def load_cifar10(batch_size, image_size=224, val_ratio=0.2):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # トレーニングセットをロード
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # トレーニングセットと検証セットに分割
    num_train = len(full_trainset)
    num_val = int(num_train * val_ratio)
    num_train -= num_val
    trainset, valset = random_split(full_trainset, [num_train, num_val])

    # データローダーを作成
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)

    # テストセットをロード
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, valloader, testloader

# 学習を行う関数
def train(opt):
    # シードの設定
    seed_everything(opt.seed)
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ハイパーパラメータの設定
    epochs = opt.epochs
    patience = opt.patience
    learning_rate = opt.learning_rate
    batch_size = opt.batch

    # データセットのロード
    train_loader, val_loader, test_loader = load_cifar10(batch_size)

    # モデルの定義
    model = VisionTransformer(
            image_size=224,
            patch_size=16,
            in_channels=3,
            embedding_dim=768,
            num_layers=12,
            num_heads=12,
            mlp_ratio=4.0,
            num_classes=10,
            hidden_dims=[64, 128, 256]
    )
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # モデルの転送
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 早期終了のためのパラメータの初期化
    val_loss_min = None
    val_loss_min_epoch = 0

    # 学習曲線のための配列の初期化
    train_losses = []
    val_losses = []

    # モデルの訓練
    for epoch in tqdm(range(epochs)):

        # 各種パラメータの初期化
        train_loss = 0.0
        val_loss = 0.0

        # モデルをtrainモードに設定
        model.train()

        # trainデータのロード
        for i, data in enumerate(train_loader, 0):
            
            # データをGPUに転送
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 勾配の初期化
            optimizer.zero_grad()

            # モデルの適用
            outputs = model(inputs)

            # 損失関数の計算
            loss = F.cross_entropy(outputs, labels)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # モデルを評価モードの設定
        model.eval()
        
        # 検証データでの評価
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                
                # データをGPUに転送
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)

                loss = F.cross_entropy(outputs, labels)
                val_loss += loss.item()
                
                
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}, Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}')
        sys.stdout.flush()

        # メモリーを最適化する
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # バリデーションロスが下がった時は結果を保存する
        if val_loss_min is None or val_loss < val_loss_min:
            model_save_directory = './latestresult'
            model_save_name = f'./latestresult/lr{learning_rate}_ep{epochs}_pa{patience}intweak.pt'
            if not os.path.exists(model_save_directory):
                os.makedirs(model_save_directory)
            torch.save(model.state_dict(), model_save_name)
            val_loss_min = val_loss
            val_loss_min_epoch = epoch
            
        # もしバリデーションロスが一定期間下がらなかったらその時点で学習を終わらせる
        elif (epoch - val_loss_min_epoch) >= patience:
            print('Early stopping due to validation loss not improving for {} epochs'.format(patience))
            break

    # テストの実行
    test_loss = []
    with torch.no_grad():
        for i, data in enumerate(test_loader,0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)

            loss = F.cross_entropy(outputs, labels)
            test_loss.append(loss.item())
        

    # テストの結果を表示
    mean_test = sum(test_loss) / len(test_loss)
    print(f'Test Loss: {mean_test:.4f}')

    # 学習プロセスをグラフ化し、保存する
    plt.figure(figsize=(15, 5))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.title("Training and Validation Loss")
    graph_save_directory = './latestresult'
    graph_save_name = f'{graph_save_directory}/lr{learning_rate}_ep{epochs}_pa{patience}intweak.png'

    if not os.path.exists(graph_save_directory):
        os.makedirs(graph_save_directory)

    plt.savefig(graph_save_name)


    return train_loss, val_loss_min

if __name__=='__main__':
    # プロセス名の設定
    setproctitle("ViTcifar10")

    # パーサーの設定
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int, required=True, help='epochs')
    parser.add_argument('--learning_rate',type=float, default=0.001, help='learning rate')
    parser.add_argument('--patience', type=int, default=5, help='patience')
    parser.add_argument('--islearnrate_search', type=str, default='false', help='is learningrate search ?')
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generators')
    parser.add_argument('--batch',type=int, default=20, help='batch size')
    opt = parser.parse_args()
    # オプションを標準出力する
    print(opt)
    sys.stdout.flush()

    # 学習率の探索を行わない場合
    if opt.islearnrate_search == 'false':
        print('-----biginning training-----')
        sys.stdout.flush()
        train_loss, val_loss = train(opt)
        print('final train loss: ',train_loss)
        print('final validation loss: ', val_loss)
        sys.stdout.flush()

    # 学習率の探索を行う場合
    elif opt.islearnrate_search == 'true':
        learning_rates = [0.0001, 0.00001, 0.001, 0.01]
        best_loss = float('inf')
        best_lr = 0
        for lr in learning_rates:
            opt.learning_rate = lr
            print(f"\nTraining with learning rate: {lr}")
            sys.stdout.flush()
            print('-----beginning training-----')
            sys.stdout.flush()
        
            train_loss, val_loss = train(opt)
        
            if val_loss < best_loss:
                best_loss = val_loss
                best_lr = lr
        print('best validation loss: ', best_loss)
        sys.stdout.flush()
        print(f"Best learning rate: {best_lr}")
        sys.stdout.flush()

    else:
        # オプションの入力が誤っている時
        print('error: inappropriate input(islearnrate_search)')

    print('-----completing training-----')