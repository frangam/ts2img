#!/home/fmgarmor/proyectos/ts2img/venv/bin/python3

import os
import numpy as np
import time
import argparse



from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.model_selection import train_test_split
# from tqdm.notebook import tqdm, trange #progress bars for jupyter notebook
from tqdm.auto import trange, tqdm #progress bars for pyhton files (not jupyter notebook)


import ts2img.activity_data as act
import ts2img.lstmimg as l



def main():
    '''Examples of runs:
    - load LOSO numpies
    * 
    $ nohup ./genimgs.py --image-tech 0 --data-name WISDM --data-folder /home/fmgarmor/proyectos/TGEN-timeseries-generation/data/ --sampling loso  > logs/lstm-img-loso.log &
    
    - Create numpies included
    $ nohup ./genimgs.py --create-numpies --data-name WISDM --data-folder /home/fmgarmor/proyectos/TGEN-timeseries-generation/data/ --sampling loso >  logs/lstm-img-loso.log &
    $ nohup ./genimgs.py --create-numpies --data-name WISDM --data-folder /home/fmgarmor/proyectos/TGEN-timeseries-generation/data/ --sampling loto >  logs/lstm-img-loto.log &
    '''
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data-name', type=str, default="WISDM", help='the database name')
    p.add_argument('--data-folder', type=str, default="/home/fmgarmor/proyectos/TGEN-timeseries-generation/data/", help='the data folder path')
    # p.add_argument('--n-folds', type=int, default=3, help='the number of k-folds')
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
    FOLDS_N = 1
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

    accs = []
    y_train_no_cat = [np.argmax(y) for y in y_train]
    p_bar_classes = tqdm(range(len(np.unique(y_train_no_cat))))
    all_classes = np.unique(y_train_no_cat)
    print("Classes available: ", all_classes)
    for i in p_bar_classes:
        y = all_classes[i]
        time.sleep(0.01) # Update Progress Bar after a while
        os.makedirs(f"{results_folder}{image_conversion}/sampling_{sampling}/holdout/train/{y}/", exist_ok=True) 
        os.makedirs(f"{results_folder}{image_conversion}/sampling_{sampling}/holdout/validation/{y}/", exist_ok=True) 
        # os.makedirs(f"{results_folder}plots/single_axis/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/train/{y}/", exist_ok=True)
        # os.makedirs(f"{results_folder}plots/single_axis/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/test/{y}/", exist_ok=True)

    from sklearn.model_selection import GroupShuffleSplit
    
    # Define the train-test split
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    # Perform the split
    train_index, val_index = next(splitter.split(X_train, y_train, groups=sj_train))
    
    # Get the train and validation data
    X_train, X_val = X_train[train_index], X_train[val_index]
    y_train, y_val = y_train[train_index], y_train[val_index]
    sj_train, sj_val = sj_train[train_index], sj_train[val_index]

    if axis_to_use >= 0:
        X_train = X_train[:,:,axis_to_use]
        X_val = X_val[:,:,axis_to_use]
    print("axis_to_use:", axis_to_use)

    print("X_train.shape", X_train.shape, "y_train.shape", y_train.shape, "sj_train.shape", sj_train.shape)
    print("X_val.shape", X_val.shape, "y_val.shape", y_val.shape, "sj_val.shape", sj_val.shape)

    if image_conversion == 0:  # Markov Transition Field
        # Create dataset for training and validation
        mtf_model = args.markov_model
        train_data_folder = f"{results_folder}{image_conversion}-reconstruction/{mtf_model}/sampling_{sampling}/holdout/train"
        val_data_folder = f"{results_folder}{image_conversion}-reconstruction/{mtf_model}/sampling_{sampling}/holdout/validation"
        trainX, trainY = l.load_or_create_dataset(train_data_folder, 0, mtf_model, X_train, y_train, create_numpies, image_conversion)
        valX, valY = l.load_or_create_dataset(val_data_folder, 0, mtf_model, X_val, y_val, create_numpies, image_conversion)
        print("After load_or_create_dataset trainX", trainX.shape, " trainY", trainY.shape, ". valX", valX.shape, " valY:", valY.shape)

        # Reshape for LSTM [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
        input_shape = (1, trainX.shape[2])
        valX = np.reshape(valX, (valX.shape[0], valX.shape[1], valX.shape[2]))

        # Load or create model for training and validation
        print("trainX after reshape", trainX.shape, ". valX", valX.shape)

        model = l.load_or_create_model(train_data_folder, f"lstm_model_holdout.keras", mtf_model, input_shape, trainX, trainY, valX, valY)
        l.save_val_results(model, valX, valY, f"{val_data_folder}/results.json")

        ts = X_val.shape[1]
        for idx in range(0, valX.shape[0], ts):
            # Extract the chunk of 129 steps
            mtf_image = valX[idx:idx + ts]
            print("mtf", mtf_image.shape)

            # Reshape mtf_image to (129, 129)
            mtf_image = mtf_image.reshape(ts, ts)
            print("MTF shape:", mtf_image.shape)

            original_series = valY[idx:idx + ts]

            print("Reconstructing series")
            print("idx", idx)
            print("class:", y_val[idx // ts])
            print("subject:", sj_val[idx // ts])
            print("Original series shape:", original_series.shape)

            l.save_image(mtf_image, f"{val_data_folder}/{y_val[idx // ts]}/", f'{sj_val[idx // ts]}_{idx // ts}.png')

            reconstructed_series = l.reconstruct_series_with_lstm(model, mtf_image, mtf_model)
            print("reconstructed_series from MTF shape:", reconstructed_series.shape)

            l.save_comparison_plot(original_series.reshape(1, -1), reconstructed_series, f"{val_data_folder}/comparison-original-reconstructed/{y_val[idx // ts]}/", f'{sj_val[idx // ts]}_{idx // ts}.png')

if __name__ == '__main__':
    main()