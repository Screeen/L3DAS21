import sys, os
import time
import json
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import Adam
import torch.utils.data as utils
from torch.utils.tensorboard import SummaryWriter
from waveunet_model.waveunet import Waveunet
from FaSNet import FaSNet_origin
import utility_functions as uf


os.environ['KMP_DUPLICATE_LIB_OK']='True'


def evaluate(model, device, criterion, num_mic, dataloader):
    model.eval()
    test_loss = 0.
    with tqdm(total=len(dataloader) // args.batch_size) as pbar, torch.no_grad():
        for example_num, (x, target) in enumerate(dataloader):
            #x, target = dyn_pad(x, target)
            target = target.to(device)
            x = x.to(device)

            outputs = model(x, num_mic)
            loss = criterion(outputs, target)
            #loss = criterion(outputs['vocals'][:,0,:], target)
            test_loss += (1. / float(example_num + 1)) * (loss - test_loss)

            pbar.set_description("Current loss: {:.4f}".format(test_loss))
            pbar.update(1)
    return test_loss

def main(args):
    if args.use_cuda:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'

    if args.fixed_seed:
        seed = 1
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    #writer = SummaryWriter(args.log_dir)
    print ('\nLoading dataset')

    #load dataset

    with open(args.training_predictors_path, 'rb') as f:
        training_predictors = pickle.load(f)
    with open(args.training_target_path, 'rb') as f:
        training_target = pickle.load(f)
    with open(args.validation_predictors_path, 'rb') as f:
        validation_predictors = pickle.load(f)
    with open(args.validation_target_path, 'rb') as f:
        validation_target = pickle.load(f)
    with open(args.test_predictors_path, 'rb') as f:
        test_predictors = pickle.load(f)
    with open(args.test_target_path, 'rb') as f:
        test_target = pickle.load(f)

    training_predictors = np.array(training_predictors)
    training_target = np.array(training_target)
    validation_predictors = np.array(validation_predictors)
    validation_target = np.array(validation_target)
    test_predictors = np.array(test_predictors)
    test_target = np.array(test_target)


    print ('\nShapes:')
    print ('Training predictors: ', training_predictors.shape)
    print ('Validation predictors: ', validation_predictors.shape)
    print ('Test predictors: ', test_predictors.shape)


    #convert to tensor
    training_predictors = torch.tensor(training_predictors).float()
    validation_predictors = torch.tensor(validation_predictors).float()
    test_predictors = torch.tensor(test_predictors).float()
    training_target = torch.tensor(training_target).float()
    validation_target = torch.tensor(validation_target).float()
    test_target = torch.tensor(test_target).float()
    #build dataset from tensors
    tr_dataset = utils.TensorDataset(training_predictors, training_target)
    val_dataset = utils.TensorDataset(validation_predictors, validation_target)
    test_dataset = utils.TensorDataset(test_predictors, test_target)
    #build data loader from dataset
    tr_data = utils.DataLoader(tr_dataset, args.batch_size, shuffle=True, pin_memory=True)
    val_data = utils.DataLoader(val_dataset, args.batch_size, shuffle=False, pin_memory=True)
    test_data = utils.DataLoader(test_dataset, args.batch_size, shuffle=False, pin_memory=True)



    #LOAD MODEL

    model = FaSNet_origin(enc_dim=64, feature_dim=64, hidden_dim=128, layer=6, segment_size=50,
                                     nspk=2, win_len=4, context_len=16, sr=16000)
    if args.use_cuda:
        print("move model to gpu")
    model = model.to(device)

    #compute number of parameters
    model_params = sum([np.prod(p.size()) for p in model.parameters()])
    print ('Total paramters: ' + str(model_params))

    # Set up the loss function
    if args.loss == "L1":
        criterion = nn.L1Loss()
    elif args.loss == "L2":
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError("Couldn't find this loss!")

    # Set up optimiser
    optimizer = Adam(params=model.parameters(), lr=args.lr)

    # Set up training state dict that will also be saved into checkpoints
    state = {"step" : 0,
             "worse_epochs" : 0,
             "epochs" : 0,
             "best_loss" : np.Inf}

    # LOAD MODEL CHECKPOINT IF DESIRED
    if args.load_model is not None:
        print("Continuing training full model from checkpoint " + str(args.load_model))
        state = uf.load_model(model, optimizer, args.load_model, args.use_cuda)


    print('TRAINING START')
    train_loss_hist = []
    val_loss_hist = []
    while state["worse_epochs"] < args.patience:
        print("Training one epoch from iteration " + str(state["step"]))
        avg_time = 0.
        model.train()
        train_loss = 0.
        with tqdm(total=len(tr_dataset) // args.batch_size) as pbar:
            #np.random.seed()
            for example_num, (x, target) in enumerate(tr_data):
                #x, target = dyn_pad(x, target)
                target = target.to(device)
                x = x.to(device)
                t = time.time()

                # Compute loss for each instrument/model
                optimizer.zero_grad()
                outputs = model(x, torch.tensor(args.num_mic).long())
                loss = criterion(outputs, target)
                #loss = criterion(outputs['vocals'][:,0,:], target)
                loss.backward()
                train_loss += (1. / float(example_num + 1)) * (loss - train_loss)
                optimizer.step()
                state["step"] += 1
                t = time.time() - t
                avg_time += (1. / float(example_num + 1)) * (t - avg_time)

                #writer.add_scalar("train_loss", loss.item(), state["step"])
                pbar.update(1)

            #PASS VALIDATION DATA
            val_loss = evaluate(model, device, criterion, torch.tensor(args.num_mic).long(), val_data)
            print("VALIDATION FINISHED: LOSS: " + str(val_loss))

            # EARLY STOPPING CHECK
            #checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_" + str(state["step"]))
            checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint")

            if val_loss >= state["best_loss"]:
                state["worse_epochs"] += 1
            else:
                print("MODEL IMPROVED ON VALIDATION SET!")
                state["worse_epochs"] = 0
                state["best_loss"] = val_loss
                state["best_checkpoint"] = checkpoint_path

                # CHECKPOINT
                print("Saving model...")
                uf.save_model(model, optimizer, state, checkpoint_path)

            state["epochs"] += 1
            #state["worse_epochs"] = 200
            train_loss_hist.append(train_loss.cpu().detach().numpy())
            val_loss_hist.append(val_loss.cpu().detach().numpy())

    #### TESTING ####
    # Test loss
    print("TESTING")
    # Load best model based on validation loss
    state = uf.load_model(model, None, state["best_checkpoint"], args.use_cuda)
    train_loss = evaluate(model, device, criterion, torch.tensor(args.num_mic).long(), tr_data)
    val_loss = evaluate(model, device, criterion, torch.tensor(args.num_mic).long(), val_data)
    test_loss = evaluate(model, device, criterion, torch.tensor(args.num_mic).long(), test_data)

    print("TEST FINISHED: LOSS: " + str(test_loss))

    results = {'train_loss': train_loss.cpu().detach().numpy(),
               'val_loss': val_loss.cpu().detach().numpy(),
               'test_loss': test_loss.cpu().detach().numpy(),
               'train_loss_hist': train_loss_hist,
               'val_loss_hist': val_loss_hist}
    print ('RESULTS')
    for i in results:
        print (i, results[i])
    out_path = os.path.join(args.results_path, 'results_dict.json')
    np.save(out_path, results)
    #writer.add_scalar("test_loss", test_loss, state["step"])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #saving parameters
    parser.add_argument('--results_path', type=str, default='RESULTS/fasnet_test')
    parser.add_argument('--checkpoint_dir', type=str, default='RESULTS/fasnet_test',
                        help='Folder to write checkpoints into')
    #dataset parameters
    parser.add_argument('--training_predictors_path', type=str, default='DATASETS/processed/task1/task1_predictors_train.pkl')
    parser.add_argument('--training_target_path', type=str, default='DATASETS/processed/task1/task1_target_train.pkl')
    parser.add_argument('--validation_predictors_path', type=str, default='DATASETS/processed/task1/task1_predictors_validation.pkl')
    parser.add_argument('--validation_target_path', type=str, default='DATASETS/processed/task1/task1_target_validation.pkl')
    parser.add_argument('--test_predictors_path', type=str, default='DATASETS/processed/task1/task1_predictors_test.pkl')
    parser.add_argument('--test_target_path', type=str, default='DATASETS/processed/task1/task1_target_test.pkl')
    #model parameters
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--num_mic', type=float, default=4)
    parser.add_argument('--use_cuda', type=str, default='True')
    parser.add_argument('--early_stopping', type=str, default='True')
    parser.add_argument('--fixed_seed', type=str, default='False')
    parser.add_argument('--instruments', type=str, nargs='+', default=["vocals"],
                        help="List of instruments to separate (default: \"vocals\")")
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of data loader worker threads (default: 1)')
    parser.add_argument('--features', type=int, default=32,
                        help='Number of feature channels per layer')
    parser.add_argument('--log_dir', type=str, default='RESULTS/waveunet/logs',
                        help='Folder to write logs into')
    parser.add_argument('--hdf_dir', type=str, default="hdf",
                        help='Dataset path')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Reload a previously trained model (whole task model)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate in LR cycle (default: 1e-3)')
    parser.add_argument('--min_lr', type=float, default=5e-5,
                        help='Minimum learning rate in LR cycle (default: 5e-5)')
    parser.add_argument('--cycles', type=int, default=2,
                        help='Number of LR cycles per epoch')
    parser.add_argument('--batch_size', type=int, default=10,
                        help="Batch size")
    parser.add_argument('--levels', type=int, default=6,
                        help="Number of DS/US blocks")
    parser.add_argument('--depth', type=int, default=1,
                        help="Number of convs per block")
    parser.add_argument('--sr', type=int, default=16000,
                        help="Sampling rate")
    parser.add_argument('--channels', type=int, default=4,
                        help="Number of input audio channels")
    parser.add_argument('--kernel_size', type=int, default=5,
                        help="Filter width of kernels. Has to be an odd number")
    parser.add_argument('--output_size', type=float, default=1.0,
                        help="Output duration")
    parser.add_argument('--strides', type=int, default=4,
                        help="Strides in Waveunet")
    parser.add_argument('--patience', type=int, default=20,
                        help="Patience for early stopping on validation set")

    parser.add_argument('--loss', type=str, default="L2",
                        help="L1 or L2")
    parser.add_argument('--conv_type', type=str, default="gn",
                        help="Type of convolution (normal, BN-normalised, GN-normalised): normal/bn/gn")
    parser.add_argument('--res', type=str, default="fixed",
                        help="Resampling strategy: fixed sinc-based lowpass filtering or learned conv layer: fixed/learned")
    parser.add_argument('--separate', type=int, default=1,
                        help="Train separate model for each source (1) or only one (0)")
    parser.add_argument('--feature_growth', type=str, default="double",
                        help="How the features in each layer should grow, either (add) the initial number of features each time, or multiply by 2 (double)")
    args = parser.parse_args()
    #eval string args
    args.use_cuda = eval(args.use_cuda)
    args.early_stopping = eval(args.early_stopping)
    args.fixed_seed = eval(args.fixed_seed)

    main(args)
