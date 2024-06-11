
# TODO: Define the params

# LSTM
hidden_size = 1000  # tamaño de las capas ocultas de LSTM
num_layers = 2  # número de capas en la LSTM
bidirectional = False  # si la LSTM es bidireccional
lstm_dropout = 0.2  # dropout para las capas de LSTM

# dataset
sequence_length = 2  # longitud de la secuencia de imágenes
batch_size = 16  # tamaño del lote para el entrenamiento

# entrenamiento
learning_rate = 0.001  # tasa de aprendizaje
epochs = 3 # número de épocas de entrenamiento
