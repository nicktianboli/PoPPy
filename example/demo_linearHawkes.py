"""
An example of traditional linear Hawkes process model without features
"""
from importlib import reload
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import dev.util as util
from model.HawkesProcess import HawkesProcessModel
from preprocess.DataIO import load_sequences_csv
from preprocess.DataOperation import data_info, EventSampler, ThinningSampler, \
    FullData, enumerate_all_events

if __name__ == '__main__':
    # hyper-parameters
    memory_size = 400
    batch_size = 128
    use_cuda = True
    use_cuda = use_cuda and torch.cuda.is_available()
    seed = 2
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


    # load event sequences from csv file
    domain_names = {'seq_id': 'seq_id',
                    'time': 'time',
                    'event': 'event'}
    database = load_sequences_csv('{}/{}/IPTV_DATA.csv'.format(util.POPPY_PATH, util.DATA_DIR),
                                  domain_names=domain_names, upperlimit=50000)
    data_info(database)

    # sample batches from database
    trainloader_thinning = DataLoader(ThinningSampler(database=database, length=memory_size),
                             batch_size=batch_size,
                             shuffle=True,
                             **kwargs)
    trainloader_interval = DataLoader(EventSampler(database=database, memorysize=memory_size),
                             batch_size=batch_size,
                             shuffle=True,
                             **kwargs)
    validationloader = DataLoader(FullData(database=database))

    #
    # initialize model
    num_type = len(database['type2idx'])
    mu_dict = {'model_name': 'NaiveExogenousIntensity',
               'parameter_set': {'activation': 'identity'}
               }
    alpha_dict = {'model_name': 'NaiveEndogenousImpact',
                  'parameter_set': {'activation': 'identity'}
                  }
    # kernel_para = np.ones((2, 1))
    kernel_para = 2*np.random.rand(2, 1)
    kernel_para[1, 0] = 2
    kernel_para = torch.from_numpy(kernel_para)
    kernel_para = kernel_para.type(torch.FloatTensor)
    kernel_dict = {'model_name': 'ExponentialKernel',
                   'parameter_set': kernel_para}
    loss_type = 'mle'
    hawkes_model = HawkesProcessModel(num_type=num_type,
                                      mu_dict=mu_dict,
                                      alpha_dict=alpha_dict,
                                      kernel_dict=kernel_dict,
                                      activation='identity',
                                      loss_type=loss_type,
                                      use_cuda=use_cuda)

    hawkes_model2 = HawkesProcessModel(num_type=num_type,
                                      mu_dict=mu_dict,
                                      alpha_dict=alpha_dict,
                                      kernel_dict=kernel_dict,
                                      activation='identity',
                                      loss_type=loss_type,
                                      use_cuda=use_cuda)

    # initialize optimizer
    optimizer_thinning = optim.SGD(hawkes_model.lambda_model.parameters(), lr=0.00000001)
    optimizer_interval = optim.SGD(hawkes_model2.lambda_model.parameters(), lr=0.00001)
    scheduler_thinning = lr_scheduler.ExponentialLR(optimizer_thinning, gamma=0.8)
    scheduler_interval = lr_scheduler.ExponentialLR(optimizer_interval, gamma=0.8)

    epochs = 10

    # train model
    hawkes_model.fit(trainloader_thinning, optimizer_thinning, epochs, scheduler=scheduler_thinning,
                     sparsity=None, nonnegative=None, use_cuda=use_cuda, validation_set=validationloader)

    hawkes_model2.fit(trainloader_interval, optimizer_interval, epochs, scheduler=scheduler_interval,
                     sparsity=None, nonnegative=None, use_cuda=use_cuda, validation_set=validationloader)
    # save model
    hawkes_model.save_model('{}/{}/full.pt'.format(util.POPPY_PATH, util.OUTPUT_DIR), mode='entire')
    hawkes_model.save_model('{}/{}/para.pt'.format(util.POPPY_PATH, util.OUTPUT_DIR), mode='parameter')

    # load model
    hawkes_model.load_model('{}/{}/full.pt'.format(util.POPPY_PATH, util.OUTPUT_DIR), mode='entire')

    # plot exogenous intensity
    all_events = enumerate_all_events(database, seq_id=1, use_cuda=use_cuda)
    hawkes_model.plot_exogenous(all_events,
                                output_name='{}/{}/exogenous_linearHawkes.png'.format(util.POPPY_PATH, util.OUTPUT_DIR))

    # plot endogenous Granger causality
    hawkes_model.plot_causality(all_events,
                                output_name='{}/{}/causality_linearHawkes.png'.format(util.POPPY_PATH, util.OUTPUT_DIR))

    # simulate new data based on trained model
    new_data, counts = hawkes_model.simulate(history=database,
                                             memory_size=memory_size,
                                             time_window=5,
                                             interval=1.0,
                                             max_number=10,
                                             use_cuda=use_cuda)
