import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.saving import register_keras_serializable

from tensorflow.keras.models import Sequential, load_model,Model
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv3D, Flatten, Dense, TimeDistributed, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

import joblib  # Para guardar y cargar datasets
import time
# from tqdm.notebook import tqdm, trange #progress bars for jupyter notebook
from tqdm.auto import trange, tqdm #progress bars for pyhton files (not jupyter notebook)
from pyts.image import MarkovTransitionField
from sklearn.preprocessing import MinMaxScaler

def normalize_data(trainX, trainY, valX, valY):
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()

    # Ajustar el escalador en los datos de entrenamiento
    trainX_reshaped = trainX.reshape(-1, trainX.shape[-1])
    valX_reshaped = valX.reshape(-1, valX.shape[-1])
    
    scaler_X.fit(trainX_reshaped)
    scaler_Y.fit(trainY.reshape(-1, 1))
    
    # Transformar los datos
    trainX = scaler_X.transform(trainX_reshaped).reshape(trainX.shape)
    valX = scaler_X.transform(valX_reshaped).reshape(valX.shape)
    
    trainY = scaler_Y.transform(trainY.reshape(-1, 1)).reshape(trainY.shape)
    valY = scaler_Y.transform(valY.reshape(-1, 1)).reshape(valY.shape)
    
    return trainX, trainY, valX, valY, scaler_X, scaler_Y

def denormalize_predictions(predictions, scaler):
    # Asegurarse de que las predicciones tengan la forma adecuada (2D)
    predictions = predictions.reshape(-1, 1)
    return scaler.inverse_transform(predictions).flatten()


# Implementar métrica R² Score
@tf.keras.utils.register_keras_serializable()
def r2_score(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())

# Implementar métrica sMAPE
@tf.keras.utils.register_keras_serializable()
def smape(y_true, y_pred):
    numerator = K.abs(y_true - y_pred)
    denominator = (K.abs(y_true) + K.abs(y_pred)) / 2.0
    return K.mean(numerator / (denominator + K.epsilon()))


def converttoimage(series, conversion_type=0):
    img = []
    if conversion_type == 0: #MarkovTransitionField
        # print("series;", series.shape)
        mtf = MarkovTransitionField(image_size=series.shape[1])
        
        # Transformar la serie temporal en un campo de transición de Markov
        markov_field = mtf.fit_transform(series)
        img = markov_field[0]
    
    # print("img shape", img.shape)
    return img

# def reconstruct_series_from_mtf(mtf_image, min_val, max_val, original_series):
#     n_states = mtf_image.shape[0]
#     n_steps = mtf_image.shape[1]
    
#     # Inicializar la serie reconstruida
#     reconstructed_series = np.zeros(n_steps)
    
#     # Mapear el primer valor original al estado más cercano
#     states = np.linspace(min_val, max_val, n_states)
#     first_value = original_series[0, 0]
#     current_state = np.argmin(np.abs(states - first_value))
#     reconstructed_series[0] = current_state
    
#     for t in range(1, n_steps):
#         # Elegir el próximo estado basado en las probabilidades de transición
#         next_state_probabilities = mtf_image[current_state]
#         next_state = np.random.choice(n_states, p=next_state_probabilities / np.sum(next_state_probabilities))
#         reconstructed_series[t] = next_state
#         current_state = next_state
    
#     # Ajustar la escala de la serie reconstruida para que coincida con la original
#     reconstructed_series = reconstructed_series * (max_val - min_val) / n_states
#     reconstructed_series += min_val
    
#     return reconstructed_series

def create_dataset_from_mtf(series_list):
    dataX, dataY = [], []

    for series in series_list:
        series_reshaped = series.reshape(1, -1)  # Reshape solo para convertir a MTF
        mtf_image = converttoimage(series_reshaped)
        
        # Vamos columna por columna
        for i in range(mtf_image.shape[1]):
            mtf_column = mtf_image[:, i].reshape(-1, 1)  # Cada entrada es una columna de la imagen MTF
            dataX.append(mtf_column)
            dataY.append(series[i])  # Valor correspondiente en la serie temporal

    dataX = np.array(dataX)
    dataY = np.array(dataY)

    # Ajustar la forma para [samples, time steps, features]
    # En este caso, queremos [samples, 1 timestep, 129 features]
    dataX = dataX.reshape((dataX.shape[0], 1, dataX.shape[1]))

    return dataX, dataY

def create_dataset_series_and_markovtransform_conv_lstm(series_list):
    dataX, dataY = [], []

    for series in series_list:
        # Generar la imagen MTF para la serie
        series_reshaped = series.reshape(1, -1)
        mtf_image = converttoimage(series_reshaped)
        
        # Ajustar la forma de la imagen MTF para agregar el canal
        mtf_image = mtf_image.reshape((mtf_image.shape[0], mtf_image.shape[1], 1))
        
        for i in range(mtf_image.shape[1]):
            mtf_slice = mtf_image[:, i].reshape(mtf_image.shape[0], 1, 1)  # Shape (129, 1, 1)

            # La forma del mtf_slice debe ser (rows, cols, channels)
            dataX.append(mtf_slice)
            dataY.append(series[i])  # Valor correspondiente en la serie temporal
    
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    print("dataX shape:", dataX.shape)  # Esperado: (1290, 129, 1, 1)
    
    # Ajustar la forma para [samples, timesteps, rows, cols, channels]
    dataX = dataX.reshape((dataX.shape[0], 1, dataX.shape[1], dataX.shape[2], dataX.shape[3]))
    
    return dataX, dataY


# def build_conv_lstm_model(input_shape):
#     inputs = Input(shape=input_shape)
    
#     # ConvLSTM layers con kernel de tamaño (1, 1)
#     x = ConvLSTM2D(filters=64, kernel_size=(1, 1), activation='relu', return_sequences=True)(inputs)
#     x = BatchNormalization()(x)
#     x = ConvLSTM2D(filters=64, kernel_size=(1, 1), activation='relu', return_sequences=True)(x)
#     x = BatchNormalization()(x)
#     x = ConvLSTM2D(filters=64, kernel_size=(1, 1), activation='relu', return_sequences=True)(x)
#     x = BatchNormalization()(x)
    
#     # Flatten y Dense layers para la predicción final
#     x = Flatten()(x)
#     x = Dense(100, activation='relu')(x)
#     x = Dense(1, activation='linear')(x)
    
#     model = Model(inputs, x)
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model

def build_conv_lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # ConvLSTM layers con kernel de tamaño (1, 1)
    x = ConvLSTM2D(filters=32, kernel_size=(1, 1), activation='relu', return_sequences=True, kernel_regularizer=l2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = ConvLSTM2D(filters=32, kernel_size=(1, 1), activation='relu', return_sequences=True, kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = ConvLSTM2D(filters=32, kernel_size=(1, 1), activation='relu', return_sequences=True, kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Flatten y Dense layers para la predicción final
    x = Flatten()(x)
    x = Dense(100, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.4)(x)
    x = Dense(1, activation='linear')(x)
    
    model = Model(inputs, x)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=[MeanAbsoluteError(), RootMeanSquaredError(), smape, r2_score])
    return model


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.4))
    model.add(LSTM(50, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='linear'))  # Activación lineal predeterminada
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=[MeanAbsoluteError(), RootMeanSquaredError(), smape, r2_score])
    return model

def build_complex_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    model.add(LSTM(100, return_sequences=True, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    model.add(LSTM(100, return_sequences=True, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    model.add(LSTM(100, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='linear'))  # Activación lineal predeterminada

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=[MeanAbsoluteError(), RootMeanSquaredError(), smape, r2_score])
    return model


def reconstruct_series_with_lstm(model, mtf_image, model_id=0):
    n_steps = mtf_image.shape[0]
    
    print(">>> Reconstructing, mtf_image shape", mtf_image.shape, " . n_steps",n_steps)
    reconstructed_series = []

    for t in range(n_steps):
        if model_id ==0 or model_id==1: #LSTMs
            input_seq = mtf_image[:, t].reshape(1, 1, -1)  # (1, 1, features)
        elif model_id == 2: #ConvLSTM
            input_seq = mtf_image[:, t].reshape(1, 1, -1, 1)
        next_value = model.predict(input_seq, verbose=0)
        reconstructed_series.append(next_value[0, 0])


    return np.array(reconstructed_series)

def plot_training_history(history, save_path, file_name):
    """
    Plots the training and validation history and saves the plot as an image.

    Parameters:
    - history: History object from the training process.
    - save_path: The directory path where the plot image will be saved.
    - file_name: The name of the plot image file.
    """
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Full path for the file
    file_path = os.path.join(save_path, file_name)
    
    # Plot training & validation loss values
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(2, 2, 2)
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(2, 2, 3)
    plt.plot(history.history['root_mean_squared_error'])
    plt.plot(history.history['val_root_mean_squared_error'])
    plt.title('Model RMSE')
    plt.ylabel('RMSE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(2, 2, 4)
    plt.plot(history.history['smape'])
    plt.plot(history.history['val_smape'])
    plt.title('Model sMAPE')
    plt.ylabel('sMAPE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()

    # Save the plot
    plt.savefig(file_path)

    # Show the plot
    plt.show()


# Guardar la imagen en un directorio
def save_image(image, directory, filename):
    """
    Guarda una imagen sin ejes ni anotaciones.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    print("Saving image at:", filepath)

    # Crear la figura y el eje con un tamaño adecuado
    fig, ax = plt.subplots(figsize=(image.shape[1], image.shape[0]))
    
    # Mostrar la imagen sin ejes ni anotaciones
    ax.imshow(image, cmap='viridis', aspect='auto')
    ax.axis('off')  # Eliminar los ejes

    # Guardar la imagen sin márgenes alrededor
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def save_comparison_plot(original_series, reconstructed_series, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    print("Saving comparison plot at:", filepath)
    
    plt.figure(figsize=(12, 6))
    plt.plot(original_series[0], label='Serie Original')
    plt.plot(reconstructed_series, label='Serie Reconstruida', linestyle='--')
    plt.legend()
    plt.title('Comparativa entre la Serie Original y la Serie Reconstruida')
    plt.xlabel('Timestamps')
    plt.ylabel('Valores')
    plt.savefig(filepath)
    plt.close()

def save_comparative_plot(series, reconstructed_series, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    plt.style.use('ggplot')  # Aplicar estilo ggplot
    plt.figure()
    plt.plot(series, label='Original Series')
    plt.plot(reconstructed_series, label='Reconstructed Series', linestyle='--')
    plt.legend()
    plt.grid(True)  # Agregar cuadrícula
    plt.tight_layout()  # Ajustar márgenes
    plt.savefig(filepath)
    plt.close()

def load_or_create_dataset(data_folder,fold, mtf_model, training_data, y_training_data, create_numpies, image_conversion):
    dataset_path = f"{data_folder}/dataset_markov_fold_{fold}.npz"
    print("dataset_path:",dataset_path)
    trainX, trainY = [],[]
    if not os.path.exists(dataset_path) or create_numpies:
        print("Creating dataset...")
        if image_conversion == 0:  # Markov
            if mtf_model == 0 or mtf_model == 1:
                trainX, trainY = create_dataset_from_mtf(training_data)
            else:
                trainX, trainY = create_dataset_series_and_markovtransform_conv_lstm(training_data)
        else:
            # Add other image conversion techniques if necessary
            pass
        os.makedirs(data_folder, exist_ok=True)
        np.savez_compressed(dataset_path, trainX=trainX, trainY=trainY)
    else:
        print("Loading dataset...")
        data = np.load(dataset_path)
        trainX = data['trainX']
        trainY = data['trainY']
    return trainX, trainY

def load_or_create_model(model_path, model_name,mtf_model, input_shape, trainX, trainY, valX, valY):
    full_model_path = f"{model_path}/{model_name}.keras"

    if not os.path.exists(full_model_path):
        os.makedirs(model_path, exist_ok=True)

        print("Training model...")
        model = []
        if mtf_model == 0: #LSTM basic
            model = build_lstm_model(input_shape)
        elif mtf_model == 1: #LSTM complex
            model = build_complex_lstm_model(input_shape)
        else:
            model = build_conv_lstm_model(input_shape)

        checkpoint = ModelCheckpoint(full_model_path, save_best_only=True, monitor='val_loss', mode='min')
        # early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')  
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, mode='min')  # Adjust learning rate
        model.summary()

        history = model.fit(trainX, trainY, validation_data=(valX, valY), epochs=200, batch_size=129, verbose=2, callbacks=[checkpoint,reduce_lr]) #, early_stopping, reduce_lr])
        plot_training_history(history, model_path+"/", 'training_plot.png')

    else:
        print("Loading model...")
        model = load_model(full_model_path)
        model.summary()


    return model

def save_val_results(model, valX, valY, file_path):
    # Evaluar el modelo en el conjunto de validación
    val_results = model.evaluate(valX, valY, verbose=0)
    print("model.metrics_names", model.metrics_names)

    if isinstance(val_results, (list, tuple)):
        val_metrics = dict(zip(model.metrics_names, val_results))
        print(f'Validation Results: {val_metrics}')
    else:
        val_metrics = {'loss': val_results}
        print(f'Validation Loss: {val_results}')

    # Guardar los resultados en un archivo JSON
    with open(file_path, 'w') as json_file:
        json.dump(val_metrics, json_file, indent=4)

def process_window(fold, dataset_folder, training_data, y_data, sj_train, TIME_STEPS=129, FOLDS_N=3, data_type="train", sampling="loso", image_conversion="default"):
    subject_samples = 0
    p_bar = tqdm(range(len(training_data)))

    for i in p_bar:
        w = training_data[i]
        sj = sj_train[i][0]
        w_y = y_data[i]
        w_y_no_cat = np.argmax(w_y)
        print("w_y", w_y, "w_y_no_cat", w_y_no_cat)
        print("w", w.shape)

        # Update Progress Bar after a while
        time.sleep(0.01)
        p_bar.set_description(f'[{data_type} | FOLD {fold} | Class {w_y_no_cat}] Subject {sj}')

        print("w sape", w.shape)


        # Iterar sobre cada variable en la serie temporal
        if len(w.shape)>1:
            for var_idx in range(w.shape[1]):
                var_series = w[:, var_idx]
                if len(var_series) == TIME_STEPS:
                    print("var_series shape || (n_samples, n_timestamps)", var_series.shape, var_series.reshape(1, -1))
                    image = converttoimage(var_series.reshape(1, -1), image_conversion)
                    # image = temporal_pattern_image(var_series.reshape(-1, 1), TIME_STEPS)
                    save_image(image, f"{dataset_folder}{image_conversion}/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/{w_y_no_cat}/", f'{sj}_{var_idx}_{subject_samples}.png')
                    reconstructed_series = image_to_series(image, len(w))
                    save_comparative_plot(var_series, reconstructed_series, f"{dataset_folder}{image_conversion}/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/{w_y_no_cat}/", f'{sj}_{var_idx}_{subject_samples}_ori_vs_rec.png')
        else:
            original_series = w.reshape(1, -1)
            print("original_series || (n_samples, n_timestamps)",original_series.shape)
            image = converttoimage(original_series, image_conversion)
            if image_conversion == 0: #Markov
                min = np.min(original_series)
                max = np.max(original_series)
                
                # reconstructed_series = reconstruct_series_with_lstm(model, initial_value, image, original_series.shape[1])
                # save_comparison_plot(w.reshape(1,-1), reconstructed_series, f"{dataset_folder}{image_conversion}-reconstruction/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/{w_y_no_cat}/", f'{sj}_-1_{subject_samples}.png')

            # image = temporal_pattern_image(w.reshape(-1, 1), TIME_STEPS)
            save_image(image, f"{dataset_folder}{image_conversion}/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/{w_y_no_cat}/", f'{sj}_-1_{subject_samples}.png')
            reconstructed_series = image_to_series(image, len(w))
            save_comparative_plot(w, reconstructed_series, f"{dataset_folder}{image_conversion}/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/{w_y_no_cat}/", f'{sj}_-1_{subject_samples}_ori_vs_rec.png')




        subject_samples += 1
    