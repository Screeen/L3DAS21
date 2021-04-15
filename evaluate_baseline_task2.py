import sys, os
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import soundfile as sf
import torch
import torch.utils.data as utils
from metrics import location_sensitive_detection
from FaSNet import FaSNet_origin
from utility_functions import load_model, save_model

'''
Load pretrained model and compute the metrics for Task 1
of the L3DAS21 challenge. The metric is: (STOI+(1-WER))/2
Command line arguments define the model parameters, the dataset to use and
where to save the obtained results.
'''


def main(args):
    if args.use_cuda:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'

    print ('\nLoading dataset')
    #LOAD DATASET
    with open(args.predictors_path, 'rb') as f:
        predictors = pickle.load(f)
    with open(args.target_path, 'rb') as f:
        target = pickle.load(f)
    predictors = np.array(predictors)
    target = np.array(target)

    print ('\nShapes:')
    print ('Predictors: ', predictors.shape)

    #convert to tensor
    predictors = torch.tensor(predictors).float()
    target = torch.tensor(target).float()
    #build dataset from tensors
    dataset_ = utils.TensorDataset(predictors, target)
    #build data loader from dataset
    dataloader = utils.DataLoader(dataset_, 1, shuffle=False, pin_memory=True)

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    #LOAD MODEL
    if args.architecture == 'seldnet':
        model = Seldnet(time_dim=args.time_dim, freq_dim=args.freq_dim, input_channels=args.input_channels,
                    output_classes=args.output_classes, pool_size=args.pool_size,
                    pool_time=args.pool_time, rnn_size=args.rnn_size, n_rnn=args.n_rnn,
                    fc_size=args.fc_size, dropout_perc=args.dropout_perc,
                    n_cnn_filters=args.n_cnn_filters, verbose=args.verbose)

    if args.use_cuda:
        print("Moving model to gpu")
    model = model.to(device)

    #load checkpoint
    state = load_model(model, None, args.model_path, args.use_cuda)

    #COMPUTING METRICS
    print("COMPUTING AVERAGE METRICS")
    print ('M: Final Task 1 metric')
    print ('W: Word Error Rate')
    print ('S: Stoi')

    TP = 0
    FP = 0
    FN = 0
    count = 0
    model.eval()
    with tqdm(total=len(dataloader) // 1) as pbar, torch.no_grad():
        for example_num, (x, target) in enumerate(dataloader):
            x = x.to(device)
            sed, doa = model(x)
            sed = sed.cpu().numpy().squeeze()
            doa = doa.cpu().numpy().squeeze()
            target = target.numpy().squeeze()
            print ('AAAAAAAAAA', sed.shape, doa.shape, target.shape)
            sys.exit(0)
            if count % args.save_sounds_freq == 0:
                sf.write(os.path.join(sounds_dir, str(example_num)+'.wav'), outputs, 16000, 'PCM_16')
                print ('metric: ', metric, 'wer: ', wer, 'stoi: ', stoi)

            else:
                print ('No voice activity on this frame')
            pbar.set_description('M:' +  str(np.round(METRIC,decimals=3)) +
                   ', W:' + str(np.round(WER,decimals=3)) + ', S: ' + str(np.round(STOI,decimals=3)))
            pbar.update(1)
            count += 1


    #visualize and save results
    results = {'word error rate': WER,
               'stoi': STOI,
               'task 1 metric': METRIC
               }

    print ('RESULTS')
    for i in results:
        print (i, results[i])
    out_path = os.path.join(args.results_path, 'task1_metrics_dict.json')
    np.save(out_path, results)

    '''
    baseline results
    word error rate 0.4957875215172642
    stoi 0.7070443256051635
    task 1 metric 0.6056284020439507
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #i/o parameters
    parser.add_argument('--model_path', type=str, default='RESULTS/Task2_test_seldnet/checkpoint')
    parser.add_argument('--results_path', type=str, default='RESULTS/Task2_test_seldnet/metrics')
    parser.add_argument('--save_sounds_freq', type=int, default=None)
    #dataset parameters
    parser.add_argument('--predictors_path', type=str, default='DATASETS/processed/task2_predictors_test.pkl')
    parser.add_argument('--target_path', type=str, default='DATASETS/processed/task2_target_test.pkl')
    parser.add_argument('--sr', type=int, default=32000)

    #model parameters
    parser.add_argument('--use_cuda', type=str, default=True)

    parser.add_argument('--architecture', type=str, default='seldnet',
                        help="model's architecture, can be vgg13, vgg16 or seldnet")
    parser.add_argument('--input_channels', type=int, default=8,
                        help="4/8 for 1/2 mics, multiply x2 if using also phase information")
    #the following parameters produce a prediction for each 100-msecs frame
    #everithing as in the original SELDNet implementation, but the time pooling and time dim
    parser.add_argument('--time_dim', type=int, default=4800)
    parser.add_argument('--freq_dim', type=int, default=256)
    parser.add_argument('--output_classes', type=int, default=14)
    parser.add_argument('--pool_size', type=str, default='[[8,2],[8,2],[2,2]]')
    parser.add_argument('--pool_time', type=str, default='True')
    parser.add_argument('--rnn_size', type=int, default=128)
    parser.add_argument('--n_rnn', type=int, default=2)
    parser.add_argument('--fc_size', type=int, default=128)
    parser.add_argument('--dropout_perc', type=float, default=0.)
    parser.add_argument('--n_cnn_filters', type=float, default=64)
    parser.add_argument('--verbose', type=str, default='False')
    parser.add_argument('--sed_loss_weight', type=float, default=1.)
    parser.add_argument('--doa_loss_weight', type=float, default=50.)


    args = parser.parse_args()
    #eval string args
    args.use_cuda = eval(args.use_cuda)
    args.pool_size= eval(args.pool_size)
    args.pool_time = eval(args.pool_time)
    args.verbose = eval(args.verbose)

    main(args)
