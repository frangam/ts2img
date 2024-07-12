#!/home/fmgarmor/proyectos/ts2img/venv/bin/python3

import os
import numpy as np
import time
import argparse



from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
# from tqdm.notebook import tqdm, trange #progress bars for jupyter notebook
from tqdm.auto import trange, tqdm #progress bars for pyhton files (not jupyter notebook)


import ts2img.activity_data as act
import ts2img.lstmimg as l



def main():
    '''Examples of runs:
    - load LOSO numpies
    * 
    $ nohup ./genimgs.py --image-tech 0 --data-name WISDM --n-folds 3 --data-folder /home/fmgarmor/proyectos/TGEN-timeseries-generation/data/ --sampling loso  > logs/lstm-img-loso.log &
    
    - Create numpies included
    $ nohup ./genimgs.py --create-numpies --data-name WISDM --n-folds 3 --data-folder /home/fmgarmor/proyectos/TGEN-timeseries-generation/data/ --sampling loso >  logs/lstm-img-loso.log &
    $ nohup ./genimgs.py --create-numpies --data-name WISDM --n-folds 3 --data-folder /home/fmgarmor/proyectos/TGEN-timeseries-generation/data/ --sampling loto >  logs/lstm-img-loto.log &
    '''
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data-name', type=str, default="WISDM", help='the database name')
    p.add_argument('--data-folder', type=str, default="/home/fmgarmor/proyectos/TGEN-timeseries-generation/data/", help='the data folder path')
    p.add_argument('--n-folds', type=int, default=3, help='the number of k-folds')
    p.add_argument('--sampling', type=str, default="loso", help='loso: leave-one-subject-out; loto: leave-one-trial-out')
    p.add_argument('--create-numpies', action="store_true", help='create numpies before; if not, load numpies')
    p.add_argument('--results-dir', type=str, default="results/plots/", help='directory to save images')
    p.add_argument('--image-tech', type=int, default=0, help='image conversion type to convert time-series to image') #0:Markov; 1:garmian; 2:garmian2; 3:rp
    p.add_argument('--select-axis', type=int, default=0, help='the axis index o -1 if using all available')
    p.add_argument('--markov-model', type=int, default=0, help='the model to predict the markov iamge')



    args = p.parse_args()
    
    axis_to_use = args.select_axis
    sampling = args.sampling
    create_numpies = args.create_numpies
    data_folder = args.data_folder
    data_name = args.data_name
    results_folder = args.results_dir
    image_conversion = args.image_tech
    FOLDS_N = args.n_folds
    TIME_STEPS, STEPS = act.get_time_setup(DATASET_NAME=data_name)
    print("TIME_STEPS", TIME_STEPS, "STEPS", STEPS)
    X_train, y_train, sj_train = None, None, None
    if not create_numpies:
        print("Loading numpies...")
        X_train, y_train, sj_train = act.load_numpy_datasets(data_name, data_folder, USE_RECONSTRUCTED_DATA=False)
    else:
        print("Creating numpies...")
        X_train, y_train, sj_train = act.create_all_numpy_datasets(data_name, data_folder, COL_SELECTED_IDXS=list(range(3, 3+3)))
        # y_train = to_categorical(y_train, dtype='uint8') 
    print("X_train", X_train.shape, "y_train", y_train.shape, "sj_train", sj_train.shape)
    


    groups = sj_train 
    if sampling == "loto":
        #TODO change 100 for an automatic extracted number greater than the max subject ID: max(sj_train)*10
        groups = [[int(sj[0])+i*100+np.argmax(y_train[i])+1] for i,sj in enumerate(sj_train)]
    print("Groups:", groups)

    sgkf = StratifiedGroupKFold(n_splits=FOLDS_N)
    accs = []
    y_train_no_cat = [np.argmax(y) for y in y_train]
    p_bar_classes = tqdm(range(len(np.unique(y_train_no_cat))))
    all_classes = np.unique(y_train_no_cat)
    print("Classes available: ", all_classes)
    for fold in range(FOLDS_N):
        for i in p_bar_classes:
            y = all_classes[i]
            time.sleep(0.01) # Update Progress Bar after a while
            os.makedirs(f"{results_folder}{image_conversion}/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/train/{y}/", exist_ok=True) 
            os.makedirs(f"{results_folder}{image_conversion}/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/test/{y}/", exist_ok=True) 
            # os.makedirs(f"{results_folder}plots/single_axis/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/train/{y}/", exist_ok=True)
            # os.makedirs(f"{results_folder}plots/single_axis/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/test/{y}/", exist_ok=True)

    for fold, (train_index, val_index) in enumerate(sgkf.split(X_train, y_train_no_cat, groups=groups)):
        training_data = X_train[train_index,:,:]
        validation_data = X_train[val_index,:,:]
        y_training_data = y_train[train_index]
        y_validation_data = y_train[val_index]
        sj_training_data = sj_train[train_index]
        sj_validation_data = sj_train[val_index]

        if axis_to_use >= 0:
            training_data = X_train[train_index,:,axis_to_use]
            validation_data = X_train[val_index,:,axis_to_use]
        print("axis_to_use:", axis_to_use)

        print("training_data.shape", training_data.shape, "y_training_data.shape", y_training_data.shape, "sj_training_data.shape", sj_training_data.shape)
        print("validation_data.shape", validation_data.shape, "y_validation_data.shape", y_validation_data.shape, "sj_validation_data.shape", sj_validation_data.shape)


        # l.process_window(fold, results_folder, training_data, y_training_data, sj_training_data, TIME_STEPS, FOLDS_N, "train", sampling, image_conversion)
        # l.process_window(fold, results_folder, validation_data, y_validation_data, sj_validation_data, TIME_STEPS,FOLDS_N, "train", sampling, image_conversion)


        if image_conversion == 0:  # Markov
            # Create dataset for the current fold
            mtf_model = args.markov_model
            fold_data_folder = f"{results_folder}{image_conversion}-reconstruction/{mtf_model}/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}"
            trainX, trainY = l.load_or_create_dataset(f"{fold_data_folder}/train",fold,mtf_model, training_data, y_training_data, create_numpies, image_conversion)
            valX, valY = l.load_or_create_dataset(f"{fold_data_folder}/validation",fold,mtf_model, validation_data, y_validation_data, create_numpies, image_conversion)
            print("After load_or_create_dataset trainX", trainX.shape, " trainY", trainY.shape, ". valX", valX.shape, " valY:", valY.shape)

            input_shape=[]
            if mtf_model == 0 or mtf_model==1: #LSTM
                # Reshape for LSTM [samples, time steps, features]
                trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
                input_shape = (1, trainX.shape[2])
                valX = np.reshape(valX, (valX.shape[0], valX.shape[1], valX.shape[2]))
            else:#ConvLSTM
                # Reshape for ConvLSTM [samples, time steps, rows, cols, channels]
                # trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1], trainX.shape[2], trainX.shape[3]))
                # valX = valX.reshape((valX.shape[0], 1, valX.shape[1], valX.shape[2], valX.shape[3]))
                input_shape = (1, trainX.shape[2], trainX.shape[3], trainX.shape[4])
            # Load or create model for the current fold
            print("trainX after reshape", trainX.shape, ". valX", valX.shape)
            
            trainX, trainY, valX, valY, scaler_X, scaler_Y = l.normalize_data(trainX, trainY, valX, valY)
            print("trainX after normalizacion", trainX.shape, ". valX", valX.shape)


            model = l.load_or_create_model(fold_data_folder, f"lstm_model_fold_{fold}.keras", mtf_model, input_shape, trainX, trainY, valX, valY)
            l.save_val_results(model, valX, valY, f"{fold_data_folder}/results.json")

            # Evaluar el modelo en el conjunto de validación

            # min_val, max_val = training_data.min(), training_data.max()
            # for idx, series in enumerate(training_data):
            #     print("idx", idx)
            #     # initial_value = series[0]
            #     print("Converting to image")
            #     mtf_image = l.converttoimage(series.reshape(1, -1))
            #     l.save_image(mtf_image, f"{fold_data_folder}/train/{y_training_data[idx]}/", f'{sj_training_data[idx]}_{idx}.png')
                
            for idx, series in enumerate(validation_data):
                print("Reconstructing series")
                print("idx", idx)
                print("class:", y_validation_data[idx])
                print("subject:", sj_validation_data[idx])
                print("val series shape:", series.shape)
                mtf_image = l.converttoimage(series.reshape(1, -1))
                l.save_image(mtf_image, f"{fold_data_folder}/validation/{y_validation_data[idx]}/", f'{sj_validation_data[idx]}_{idx}.png')

                reconstructed_series = l.reconstruct_series_with_lstm(model, mtf_image, mtf_model)
                
                # Desnormalizar la serie reconstruida
                reconstructed_series = l.denormalize_predictions(reconstructed_series, scaler_Y)
    
                
                l.save_comparison_plot(series.reshape(1, -1), reconstructed_series, f"{fold_data_folder}/comparison-original-reconstructed/{y_validation_data[idx]}/", f'{sj_validation_data[idx]}_{idx}.png')


if __name__ == '__main__':
    main()