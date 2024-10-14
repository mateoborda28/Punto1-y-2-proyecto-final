import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pycaret.classification import predict_model
import tempfile

## Convolución ###
# Definir los kernels para detección de bordes
kernel_horizontal = np.array([[1, 1, 1], [0, 0, 0], [-1,-1,-1]])
kernel_vertical = np.array([[1, 0,-1], [1, 0,-1], [1, 0,-1]])

def apply_convolution(matrix, kernel_type="horizontal"):
    # Seleccionar el kernel
    if kernel_type == "horizontal":
        kernel = kernel_horizontal
    elif kernel_type == "vertical":
        kernel = kernel_vertical
    else:
        raise ValueError("El tipo de kernel debe ser 'horizontal' o 'vertical'.")

    # Inicializar la matriz de salida
    output = np.zeros((matrix.shape[0] - 2, matrix.shape[1] - 2))

    # Realizar la convolución
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            # Extraer la submatriz de 3x3
            sub_matrix = matrix[i:i+3, j:j+3]
            # Calcular la suma de la multiplicación del kernel con la submatriz
            output[i, j] = np.sum(sub_matrix * kernel)
    
    return output

### Padding ####

def add_padding(matrix, padding_rows, padding_cols):
    # Asegurarse de que la matriz es un array de numpy
    matrix = np.array(matrix)
    
    # Crear matrices de ceros con el tamaño adecuado
    row_padding = np.zeros((padding_rows, matrix.shape[1]))  # Padding de filas
    col_padding = np.zeros((matrix.shape[0] + padding_rows * 2, padding_cols))  # Padding de columnas
    
    # Concatenar las filas de ceros en la parte superior y en la parte inferior
    padded_matrix = np.vstack((row_padding, matrix, row_padding))
    
    # Concatenar las columnas de ceros a la izquierda y a la derecha
    padded_matrix = np.hstack((col_padding, padded_matrix, col_padding))
    
    return padded_matrix

### Stride ###

def apply_stride(matrix, kernel_type, stride):
    # Asegurarse de que la matriz es un array de numpy
    matrix = np.array(matrix)

    # Definir el kernel para la detección de bordes
    if kernel_type == "horizontal":
        kernel = np.array([[1, 1, 1], 
                           [0, 0, 0], 
                           [-1, -1, -1]])
    elif kernel_type == "vertical":
        kernel = np.array([[1, 0, -1], 
                           [1, 0, -1], 
                           [1, 0, -1]])
    else:
        raise ValueError("Tipo de kernel no válido. Usa 'horizontal' o 'vertical'.")

    # Dimensiones de la matriz original y de salida
    output_rows = (matrix.shape[0] - 2) // stride + 1
    output_cols = (matrix.shape[1] - 2) // stride + 1
    output = np.zeros((output_rows, output_cols))

    # Aplicar el kernel con el stride
    for i in range(0, matrix.shape[0] - 2, stride):
        for j in range(0, matrix.shape[1] - 2, stride):
            # Extraer la submatriz de 3x3
            sub_matrix = matrix[i:i + 3, j:j + 3]
            # Calcular la suma de la multiplicación del kernel con la submatriz
            output[i // stride, j // stride] = np.sum(sub_matrix * kernel)

    return output

### Stacking ###

def generate_kernels(n):
    # Genera n kernels aleatorios de 3x3
    return [np.random.randint(-1, 2, (3, 3)) for _ in range(n)]

def apply_convolution_with_kernel(matrix, kernel):
    # Aplica convolución con el kernel dado
    output = np.zeros((matrix.shape[0] - 2, matrix.shape[1] - 2))
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            sub_matrix = matrix[i:i+3, j:j+3]
            output[i, j] = np.sum(sub_matrix * kernel)
    return output

def generate_feature_maps(matrix, n):
    # Genera n kernels y aplica convolución en horizontal y vertical para cada uno
    kernels = generate_kernels(n)
    feature_maps = []

    for kernel in kernels:
     horizontal_map = apply_convolution_with_kernel(matrix, kernel)  # Cambiado a apply_convolution_with_kernel
     vertical_map = apply_convolution_with_kernel(matrix, kernel.T)  # Transpuesta para detección vertical
     feature_maps.append((horizontal_map, vertical_map))  # Agregamos como un par
    
    return feature_maps

### Maxpooling ###

def max_pooling(matrix, stride=1):
    # Dimensiones de la matriz original
    rows, cols = matrix.shape
    
    # Calcular las dimensiones de la matriz de salida
    output_rows = (rows - 2) // stride + 1
    output_cols = (cols - 2) // stride + 1
    
    # Crear la matriz de salida
    output = np.zeros((output_rows, output_cols))
    
    # Recorrer la matriz con el kernel de 2x2
    for i in range(0, rows - 1, stride):
        for j in range(0, cols - 1, stride):
            # Extraer el bloque de 2x2
            block = matrix[i:i+2, j:j+2]
            # Calcular el valor máximo en el bloque
            output[i // stride, j // stride] = np.max(block)
    
    return output

# Título de la API
st.title("Predicción pixeles")

# Botón para subir archivo Excel
uploaded_file = st.file_uploader("Cargar archivo Excel", type=["xlsx"])
if uploaded_file is not None:      
     try:       # Leer la matriz desde el archivo Excel
      df = pd.read_excel(uploaded_file, header =None)
      matrix = df.to_numpy()  # Convertir el DataFrame en una matriz numpy
      option = st.selectbox("Selecciona la operación:", ["Convolución", "Padding", "Stride", "Stacking", "Max Pooling"])
    # Menú de opciones
      if option == "Convolución":
            kernel_type = st.selectbox("Selecciona tipo de kernel:", ["horizontal", "vertical"])
            if st.button("Calcular"):
             result = apply_convolution(matrix, kernel_type)
             st.write("Resultado de la convolución:")
             st.write(result)

      elif option == "Padding":       
            padding_rows = st.number_input("Ingrese la cantidad de filas a agregar:", min_value=0, value=0)
            padding_cols = st.number_input("Ingrese la cantidad de columnas a agregar:", min_value=0, value=0)
            if st.button("Calcular"):
             result = add_padding(matrix, padding_rows, padding_cols)
             st.write("Resultado del padding:")
             st.write(result)

      elif option == "Stride":
            kernel_type = st.selectbox("Selecciona tipo de kernel:", ("horizontal", "vertical"))
            stride = st.number_input("Ingrese el valor del stride:", min_value=1, value=1)
            if st.button("Calcular"):
             result = apply_stride(matrix, kernel_type, stride)
             st.write("Resultado con stride:")
             st.write(result)

      elif option == "Stacking":
         n = st.number_input("Número de kernels aleatorios:", min_value=1, value=5)
         if st.button("Calcular"):
          feature_maps = generate_feature_maps(matrix, n)
          st.write("Resultados de las convoluciones con kernels aleatorios:")
          for idx, (horizontal_map, vertical_map) in enumerate(feature_maps):
            st.write(f"Kernels {idx + 1} (Horizontal):")
            st.write(horizontal_map)
            st.write(f"Kernels {idx + 1} (Vertical):")
            st.write(vertical_map)


      elif option == "Max Pooling":
        stride = st.number_input("Ingrese el valor del stride:", min_value=1, value=1)
        if st.button("Calcular"):
            result = max_pooling(matrix, stride)
            st.write("Resultado de Max Pooling:")
            st.write(result)
    
     except Exception as e:  # Captura de excepciones
            st.error(f"Error al cargar el archivo: {e}")