import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from dataset import ContaminationDataset, get_train_test_data
from net_model import netModel

atribute = 'NO2'
device = 'cpu'
BATCH_SIZE = 128
PERCENT_TEST = 0.05
EPOCH = 20
name_file = f'neural_model/models/{atribute}_model.pth'
LEARNING_RATE = 0.001
GAMMA = 0.85
INPUT_SIZE = 9


def train_model(model, train_loader, test_set, optimizer, criterion, epoch):
    model.to(device)
    model.train()
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)
    for epoch in range(epoch):
        for i, data in enumerate(train_loader, 0):
            out = data[-1].float().to(device)
            input_train = list()
            for element in data[:-1]:
                input_train.append(element.float().to(device))
            optimizer.zero_grad()
            output_model = model(input_train)
            out = out.view(out.shape[0], 1)

            loss = criterion(output_model, out)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'Epoca:{epoch}, Step:{i}, Perdida: {loss.sum()}')
        scheduler.step()

    print("Entrenamiento finalizado, Evaluando...")
    evaluate_model(model, test_set)
    return model


def evaluate_model(model, test_set, number_show=50):
    model.eval()
    loss_list = []
    mean_out = []
    mean_real = []
    with torch.no_grad():
        for i, name in enumerate(test_set.keys(), 0):
            print(f'Para la estacion: {name}')
            out_list = []
            real_out_list = []
            mean_real_name = []
            mean_out_name = []
            data_loader = torch.utils.data.DataLoader(test_set[name], batch_size=1, shuffle=False, num_workers=2)
            for j, data in enumerate(data_loader, 0):
                out = data[-1].float().to(device)
                input_train = list()
                for element in data[:-1]:
                    input_train.append(element.float().to(device))
                output_model = model(input_train)
                out = out.view(out.shape[0], 1)
                mean_real_name.append(out.item())
                mean_out_name.append(output_model.item())
                loss_list.append(abs(output_model.item() - out.item()))
                out_list.append(output_model.item())
                real_out_list.append(out.item())
                if j < number_show:
                    print(
                        f'Numero:{j}, salida obtenida: {output_model.item()}, salida real: {out.item()}')
                if j == number_show:
                    plt.figure(i)
                    plt.title(f'Salidas reales y salidas de la red para {name}')
                    plt.plot(range(0, number_show+1), out_list, '-o', label='red')
                    plt.plot(range(0, number_show+1), real_out_list, '-o', label='real')
                    plt.legend(loc='upper right')
            mean_out_name = np.array(mean_out_name)
            mean_real_name = np.array(mean_real_name)
            mean_out.append(np.mean(mean_out_name))
            mean_real.append(np.mean(mean_real_name))
    loss_list = np.array(loss_list)
    print(f'Diferencia media entre salida real y salida de la red: {np.mean(loss_list)}')
    plt.figure(len(test_set)+1)
    plt.title(f'Salidas reales y de la red para la media de todas las estaciones')
    plt.plot(range(0, len(test_set)), mean_out, '-o', label='red')
    plt.plot(range(0, len(test_set)), mean_real, '-o', label='real')
    plt.legend(loc='upper right')
    plt.show()


print('Cargando modelo...')
model = netModel(input_size=INPUT_SIZE)
print('Cargando dataset...')
train_set, test_set = get_train_test_data(atribute, PERCENT_TEST, 10)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print('Entrenamiento [1], Evaluacion [2]')
option = input('Seleccione opcion: ')

if option == '1':
    model = train_model(model, train_loader, test_set, optimizer, criterion, epoch=EPOCH)
    torch.save(model.state_dict(), name_file)
else:
    model.load_state_dict(torch.load(name_file))
    evaluate_model(model, test_set)
