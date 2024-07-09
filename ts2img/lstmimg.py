import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import joblib  # Para guardar y cargar datasets
import time
# from tqdm.notebook import tqdm, trange #progress bars for jupyter notebook
from tqdm.auto import trange, tqdm #progress bars for pyhton files (not jupyter notebook)
from pyts.image import MarkovTransitionField

def converttoimage(series, conversion_type=0):
    img = []
    if conversion_type == 0: #MarkovTransitionField
        print("series;", series.shape)
        mtf = MarkovTransitionField(image_size=series.shape[1])
        
        # Transformar la serie temporal en un campo de transición de Markov
        markov_field = mtf.fit_transform(series)
        img = markov_field[0]
    
    print("img shape", img.shape)
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

def create_dataset_series_and_markovtransform(series_list, look_back=1):
    dataX, dataY = [], []
    for series in series_list:
        series_reshaped = series.reshape(1, -1)  # Reshape solo para convertir a MTF
        mtf_image = converttoimage(series_reshaped)
        n_states = mtf_image.shape[0]
        states = np.linspace(series.min(), series.max(), n_states)
        for i in range(len(series) - look_back):
            a = series[i:(i + look_back)]
            mtf_slice = mtf_image[:, i:(i + look_back)]
            a_with_mtf = np.concatenate([a.reshape(-1, 1), mtf_slice.T], axis=1)
            dataX.append(a_with_mtf)
            dataY.append(series[i + look_back])
    return np.array(dataX), np.array(dataY)


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def reconstruct_series_with_lstm(model, initial_value, mtf_image, n_steps):
    n_states = mtf_image.shape[0]
    states = np.linspace(initial_value.min(), initial_value.max(), n_states)
    
    # Mapear el primer valor original al estado más cercano
    first_value = initial_value
    current_state = np.argmin(np.abs(states - first_value))
    
    input_seq = np.array([[initial_value, *mtf_image[:, 0]]]).reshape((1, 1, -1))
    reconstructed_series = [initial_value]
    
    for t in range(1, n_steps):
        next_value = model.predict(input_seq)
        reconstructed_series.append(next_value[0, 0])
        
        if t < n_steps - 1:
            next_mtf_slice = mtf_image[:, t].reshape(1, -1)
            input_seq = np.concatenate([next_value, next_mtf_slice], axis=1).reshape((1, 1, -1))
    
    return np.array(reconstructed_series)





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

def load_or_create_dataset(data_name, data_folder, create_numpies, image_conversion, look_back=1):
    dataset_path = f"{data_folder}/{data_name}_dataset.npz"
    if not os.path.exists(dataset_path) or create_numpies:
        print("Creating dataset...")
        num_series = 5448
        series_length = 129
        all_series = np.random.rand(num_series, series_length)
        if image_conversion == 0:  # Markov
            trainX, trainY = create_dataset_series_and_markovtransform(all_series, look_back)
        else:
            # Add other image conversion techniques if necessary
            pass
        np.savez_compressed(dataset_path, trainX=trainX, trainY=trainY, all_series=all_series)
    else:
        print("Loading dataset...")
        data = np.load(dataset_path)
        trainX = data['trainX']
        trainY = data['trainY']
        all_series = data['all_series']
    return trainX, trainY, all_series

def load_or_create_model(model_path, input_shape, trainX, trainY):
    if not os.path.exists(model_path):
        print("Training model...")
        model = build_lstm_model(input_shape)
        checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='loss', mode='min')
        early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min')
        model.fit(trainX, trainY, epochs=50, batch_size=64, verbose=2, callbacks=[checkpoint, early_stopping])
    else:
        print("Loading model...")
        model = load_model(model_path)
    return model

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
    