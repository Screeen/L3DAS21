import argparse
import os, sys
import numpy as np
import librosa
import pickle
import random
import utility_functions as uf
import yaml
import logging

logger = logging.getLogger(__name__)

from dcase2019 import cls_feature_class
from dcase2019 import utils

'''
Process the unzipped dataset folders and output numpy matrices (.pkl files)
containing the pre-processed data for task1 and task2, separately.
Separate training, validation and test matrices are saved.
Command line inputs define which task to process and its parameters.
'''

sound_classes_dict_task2 = {'Chink_and_clink':0,
                           'Computer_keyboard':1,
                           'Cupboard_open_or_close':2,
                           'Drawer_open_or_close':3,
                           'Female_speech_and_woman_speaking':4,
                           'Finger_snapping':5,
                           'Keys_jangling':6,
                           'Knock':7,
                           'Laughter':8,
                           'Male_speech_and_man_speaking':9,
                           'Printer':10,
                           'Scissors':11,
                           'Telephone':12,
                           'Writing':13}

def preprocessing_task1(args):
    '''
    predictors output: ambisonics mixture waveforms
                       Matrix shape: -x: data points
                                     -4 or 8: ambisonics channels
                                     -signal samples

    target output: monoaural clean speech waveforms
                   Matrix shape: -x: data points
                                 -1: it's monoaural
                                 -signal samples
    '''
    sr_task1 = 16000

    def pad(x, size=sr_task1*10):
        #pad all sounds to 10 seconds
        length = x.shape[-1]
        if length > size:
            pad = x[:,:size]
        else:
            pad = np.zeros((x.shape[0], size))
            pad[:,:length] = x
        return pad

    def process_folder(folder, args):
        #process single dataset folder
        print ('Processing ' + folder + ' folder...')
        predictors = []
        target = []
        count = 0
        main_folder = os.path.join(args.input_path, folder)
        contents = os.listdir(main_folder)
        for sub in contents:
            sub_folder = os.path.join(main_folder, sub)
            contents_sub = os.listdir(sub_folder)
            for lower in contents_sub:
                lower_folder = os.path.join(sub_folder, lower)
                data_path = os.path.join(lower_folder, 'data')
                data = os.listdir(data_path)
                data = [i for i in data if i.split('.')[0].split('_')[-1]=='A']  #filter files with mic B
                for sound in data:
                    sound_path = os.path.join(data_path, sound)
                    target_path = '/'.join((sound_path.split('/')[:-2] + ['labels'] + [sound_path.split('/')[-1]]))  #change data with labels
                    target_path = target_path[:-6] + target_path[-4:]  #remove mic ID
                    #target_path = sound_path.replace('data', 'labels').replace('_A', '')  #old wrong line
                    samples, sr = librosa.load(sound_path, sr_task1, mono=False)
                    #samples = pad(samples)
                    if args.num_mics == 2:  # if both ambisonics mics are wanted
                        #stack the additional 4 channels to get a (8, samples) shap
                        B_sound_path = sound_path[:-5] + 'B' +  sound_path[-4:]  #change A with B
                        #B_sound_path = sound_path.replace('A', 'B')  #old
                        samples_B, sr = librosa.load(B_sound_path, sr_task1, mono=False)
                        #samples_B = pad(samples_B)
                        samples = np.concatenate((samples,samples_B), axis=-2)

                    samples_target, sr = librosa.load(target_path, sr_task1, mono=False)
                    samples_target = samples_target.reshape((1, samples_target.shape[0]))
                    #samples_target = pad(samples_target)
                    #append to final arrays

                    if args.segmentation_len is not None:
                        #segment longer file to shorter frames
                        #not padding if segmenting to avoid silence frames
                        segmentation_len_samps = int(sr_task1 * args.segmentation_len)
                        predictors_cuts, target_cuts = uf.segment_waveforms(samples, samples_target, segmentation_len_samps)
                        for i in range(len(predictors_cuts)):
                            predictors.append(predictors_cuts[i])
                            target.append(target_cuts[i])
                            #print (predictors_cuts[i].shape, target_cuts[i].shape)
                    else:
                        samples = pad(samples)
                        samples_target = pad(samples_target)
                        predictors.append(samples)
                        target.append(samples_target)
                    count += 1
                    if args.num_data is not None and count >= args.num_data:
                        break
                else:
                    continue
                break
            else:
                continue
            break

        return predictors, target

    #process all required folders
    predictors_test, target_test = process_folder('L3DAS_Task1_dev', args)
    if args.training_set == 'train100':
        predictors_train, target_train = process_folder('L3DAS_Task1_train100', args)
    elif args.training_set == 'train360':
        predictors_train, target_train = process_folder('L3DAS_Task1_train360', args)
    elif args.training_set == 'both':
        predictors_train100, target_train100 = process_folder('L3DAS_Task1_train100')
        predictors_train360, target_train360 = process_folder('L3DAS_Task1_train360')
        predictors_train = predictors_train100 + predictors_train360
        target_train = target_train100 + target_train360

    #split train set into train and development
    split_point = int(len(predictors_train) * args.train_val_split)
    predictors_training = predictors_train[:split_point]    #attention: changed training names
    target_training = target_train[:split_point]
    predictors_validation = predictors_train[split_point:]
    target_validation = target_train[split_point:]

    #save numpy matrices in pickle files
    print ('Saving files')
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    with open(os.path.join(args.output_path,'task1_predictors_train.pkl'), 'wb') as f:
        pickle.dump(predictors_training, f, protocol=4)
    with open(os.path.join(args.output_path,'task1_predictors_validation.pkl'), 'wb') as f:
        pickle.dump(predictors_validation, f, protocol=4)
    with open(os.path.join(args.output_path,'task1_predictors_test.pkl'), 'wb') as f:
        pickle.dump(predictors_test, f, protocol=4)
    with open(os.path.join(args.output_path,'task1_target_train.pkl'), 'wb') as f:
        pickle.dump(target_training, f, protocol=4)
    with open(os.path.join(args.output_path,'task1_target_validation.pkl'), 'wb') as f:
        pickle.dump(target_validation, f, protocol=4)
    with open(os.path.join(args.output_path,'task1_target_test.pkl'), 'wb') as f:
        pickle.dump(target_test, f, protocol=4)

    if args.segmentation_len is not None:
        #if segmenting, generate also a test set matrix without segmenting, just for the evaluation
        args.segmentation_len = None
        print ('processing uncut test set')
        predictors_test_uncut, target_test_uncut = process_folder('L3DAS_Task1_dev', args)
        print ('Saving files')
        with open(os.path.join(args.output_path,'task1_predictors_test_uncut.pkl'), 'wb') as f:
            pickle.dump(predictors_test_uncut, f)
        with open(os.path.join(args.output_path,'task1_target_test_uncut.pkl'), 'wb') as f:
            pickle.dump(target_test_uncut, f)

    print ('Matrices successfully saved')
    print ('Training set shape: ', np.array(predictors_training).shape, np.array(target_training).shape)
    print ('Validation set shape: ', np.array(predictors_validation).shape, np.array(target_validation).shape)
    print ('Test set shape: ', np.array(predictors_test).shape, np.array(target_test).shape)


def preprocessing_task2(args):
    '''
    predictors output: ambisonics stft
                       Matrix shape: -x data points
                                     - num freqency bins
                                     - num time frames
    target output: matrix containing all active sounds and their position at each
                   100msec frame.
                   Matrix shape: -x data points
                                 -600: frames
                                 -168: 14 (clases) * 3 (max simultaneous sounds per frame)
                                       concatenated to 14 (classes) * 3 (max simultaneous sounds per frame) * 3 (xyz coordinates)
    '''
    sr_task2 = 32000
    sound_classes=['Chink_and_clink','Computer_keyboard','Cupboard_open_or_close',
             'Drawer_open_or_close','Female_speech_and_woman_speaking',
             'Finger_snapping','Keys_jangling','Knock',
             'Laughter','Male_speech_and_man_speaking',
             'Printer','Scissors','Telephone','Writing']
    file_size=60.0
    max_label_distance = 2.  #maximum xyz value (serves for normalization)

    def process_folder(folder, args):
        print ('Processing ' + folder + ' folder...')
        predictors = []
        target = []
        data_path = os.path.join(folder, 'data')
        labels_path = os.path.join(folder, 'labels')

        data = os.listdir(data_path)
        data = [i for i in data if i.split('.')[0].split('_')[-1]=='A']
        count = 0
        for sound in data:
            ov_set = sound.split('_')[-3]
            if ov_set in args.ov_subsets:  #if data point is in the desired subsets ov
                target_name = 'label_' + sound.replace('_A', '').replace('.wav', '.csv')
                sound_path = os.path.join(data_path, sound)
                target_path = os.path.join(data_path, target_name)
                target_path = '/'.join((target_path.split('/')[:-2] + ['labels'] + [target_path.split('/')[-1]]))  #change data with labels
                #target_path = target_path.replace('data', 'labels')  #old
                samples, sr = librosa.load(sound_path, sr_task2, mono=False)
                if args.num_mics == 2:  # if both ambisonics mics are wanted
                    #stack the additional 4 channels to get a (8, samples) shape
                    B_sound_path = sound_path[:-5] + 'B' +  sound_path[-4:]  #change A with B
                    #B_sound_path = sound_path.replace('A', 'B')  old
                    samples_B, sr = librosa.load(B_sound_path, sr_task2, mono=False)
                    samples = np.concatenate((samples,samples_B), axis=-2)

                #compute stft

                stft = uf.spectrum_fast(samples, nperseg=args.stft_nperseg,
                                        noverlap=args.stft_noverlap,
                                        window=args.stft_window,
                                        output_phase=args.output_phase)

                #stft = np.reshape(samples, (samples.shape[1], samples.shape[0],
                #                     samples.shape[2]))


                #compute matrix label
                label = uf.csv_to_matrix_task2(target_path, sound_classes_dict_task2,
                                               dur=60, step=args.frame_len/1000., max_loc_value=2.,
                                               no_overlaps=args.no_overlaps)  #eric func

                #label = uf.get_label_task2(target_path,0.1,file_size,sr_task2,          #giuseppe func
                #                        sound_classes,int(file_size/(args.frame_len/1000.)),
                #                        max_label_distance)


                #segment into shorter frames
                if args.predictors_len_segment is not None and args.target_len_segment is not None:
                    #segment longer file to shorter frames
                    #not padding if segmenting to avoid silence frames
                    predictors_cuts, target_cuts = uf.segment_task2(stft, label, predictors_len_segment=args.predictors_len_segment,
                                                    target_len_segment=args.target_len_segment, overlap=args.segment_overlap)

                    for i in range(len(predictors_cuts)):
                        predictors.append(predictors_cuts[i])
                        target.append(target_cuts[i])
                        #print (predictors_cuts[i].shape, target_cuts[i].shape)
                else:

                    predictors.append(stft)
                    target.append(label)

                #print (samples.shape, np.max(label), np.min(label))

                count += 1
                if args.num_data is not None and count >= args.num_data:
                    break


        return predictors, target

    train_folder = os.path.join(args.input_path, 'L3DAS_Task2_train')
    test_folder = os.path.join(args.input_path, 'L3DAS_Task2_dev')

    predictors_train, target_train = process_folder(train_folder, args)
    predictors_test, target_test = process_folder(test_folder, args)

    predictors_test = np.array(predictors_test)
    target_test = np.array(target_test)
    #print (predictors_test.shape, target_test.shape)

    #split train set into train and development
    split_point = int(len(predictors_train) * args.train_val_split)
    predictors_training = predictors_train[:split_point]    #attention: changed training names
    target_training = target_train[:split_point]
    predictors_validation = predictors_train[split_point:]
    target_validation = target_train[split_point:]

    #save numpy matrices into pickle files
    print ('Saving files')
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    with open(os.path.join(args.output_path,'task2_predictors_train.pkl'), 'wb') as f:
        pickle.dump(predictors_training, f, protocol=4)
    with open(os.path.join(args.output_path,'task2_predictors_validation.pkl'), 'wb') as f:
        pickle.dump(predictors_validation, f, protocol=4)
    with open(os.path.join(args.output_path,'task2_predictors_test.pkl'), 'wb') as f:
        pickle.dump(predictors_test, f, protocol=4)
    with open(os.path.join(args.output_path,'task2_target_train.pkl'), 'wb') as f:
        pickle.dump(target_training, f, protocol=4)
    with open(os.path.join(args.output_path,'task2_target_validation.pkl'), 'wb') as f:
        pickle.dump(target_validation, f, protocol=4)
    with open(os.path.join(args.output_path,'task2_target_test.pkl'), 'wb') as f:
        pickle.dump(target_test, f, protocol=4)

    print ('Matrices successfully saved')
    print ('Training set shape: ', np.array(predictors_training).shape, np.array(target_training).shape)
    print ('Validation set shape: ', np.array(predictors_validation).shape, np.array(target_validation).shape)
    print ('Test set shape: ', np.array(predictors_test).shape, np.array(target_test).shape)

def batch_feature_extraction_dcase2019(dataset_name):
    # Extracts the features, labels, and normalizes the training and test split features. Make sure you update the location
    # of the downloaded datasets before in the cls_feature_class.py

    overlaps = [ii + 1 for (ii, el) in enumerate(conf["ov_subsets"])]

    # Extracts feature and labels for all overlap and splits
    for ovo in overlaps:  # Change to [1] if you are only calculating the features for overlap 1.
        for splito in [2, 3]:  #  Change to [1] if you are only calculating features for split 1.
            for nffto in [conf["stft_nperseg"]]:  # use 512 point FFT.
                feat_cls = cls_feature_class.FeatureClass(ov=ovo, split=splito, nfft=nffto, dataset=dataset_name)

                # Extract features and normalize them
                feat_cls.extract_all_feature()
                feat_cls.preprocess_features()

                # # Extract labels in regression mode
                feat_cls.extract_all_labels('regr', 0)

import cfg
if __name__ == '__main__':

    cfg.init()
    utils.setup_logger(os.path.curdir)

    args = parser_reader()
    default_conf = config_reader()
    # Merge argparse and default configuration, giving priority to the ARGPARSE.
    # Do not update parameters whose value is "None"
    cfg.conf.update(default_conf)
    cfg.conf.update((k, v) for k, v in vars(args).items() if v is not None)

    conf = cfg.conf
    if conf["dataset_format"] == "l3das2021":
        if conf.task == 1:
            preprocessing_task1(conf)
        elif conf.task == 2:
            preprocessing_task2(conf)
    elif conf["dataset_format"] == "dcase2019":
        batch_feature_extraction_dcase2019(conf["dataset_name"])
