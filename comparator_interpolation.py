import torch
from neural_model.dataset import get_train_test_data
from neural_model.net_model import netModel, netModel_I
from utils import compare_interpolation_with_model, compare_interpolation_with_model_stadistics

PERCENT_TEST = 0.15
atribute = 'O3'
name_file = f'neural_model/models/{atribute}_model.pth'
name_file_i = f'neural_model/models/{atribute}_model_inter-2.pth'
model = netModel(input_size=10)
model.load_state_dict(torch.load(name_file))
model_i = netModel_I(input_size=10)
model_i.load_state_dict(torch.load(name_file_i))
train_set, test_set = get_train_test_data(atribute, PERCENT_TEST, 10)

name = test_set.keys()
compare_interpolation_with_model_stadistics(model, model_i, name, atribute)
compare_interpolation_with_model(model, model_i, name, ['02-10-2018', '07-18-2018', '12-01-2016', '04-15-2017', '10-15-2015'], atribute)
