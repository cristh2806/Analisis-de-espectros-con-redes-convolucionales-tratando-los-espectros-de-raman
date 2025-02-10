"""
Análisis de Espectros Raman para Detección de Cáncer
Equipo: [PyoneersHub]
Fecha: [10/02/2025]
"""

# %% --------------------------- 
# 1. CONFIGURACIÓN INICIAL
# ------------------------------
import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import shap
import psutil
from datetime import datetime
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from kerastuner.tuners import BayesianOptimization
from flask import Flask, request, jsonify
from flask_swagger import swagger
from flask_swagger_ui import get_swaggerui_blueprint
import tkinter as tk
from tkinter import filedialog, messagebox
from threading import Thread

# %% ---------------------------
# 2. CONFIGURACIÓN DE LOGGING
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# %% ---------------------------
# 3. GENERACIÓN DE DATOS REALISTAS
# ------------------------------
class DataGenerator:
    @staticmethod
    def generate_spectra(num_samples: int = 1000, num_wavelengths: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera espectros Raman sintéticos con características realistas
        """
        X = np.zeros((num_samples, num_wavelengths))
        y = np.random.randint(0, 2, num_samples)
        
        for i in range(num_samples):
            # Generar picos característicos
            n_peaks = np.random.randint(3, 6)
            peaks = np.random.choice(np.arange(500, 900), n_peaks, replace=False)
            intensities = np.random.uniform(1, 5, n_peaks)
            
            # Añadir ruido y baseline
            X[i] = np.random.normal(0, 0.1, num_wavelengths)
            X[i, peaks] += intensities
            X[i] = np.convolve(X[i], np.ones(5)/5, mode='same')
            
            # Diferenciar muestras cancerosas
            if y[i] == 1:
                X[i, peaks] *= 1.5
                X[i] += np.random.normal(0, 0.2, num_wavelengths)
        
        return X, y

# %% ---------------------------
# 4. PREPROCESAMIENTO DE DATOS
# ------------------------------
class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def preprocess(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> tuple:
        """
        Preprocesamiento completo con validación de datos
        """
        # Validación de dimensiones
        if X.shape[1] != 1000:
            raise ValueError("Los espectros deben tener exactamente 1000 características")
        
        # Normalización
        if not self.is_fitted:
            X_scaled = self.scaler.fit_transform(X)
            self.is_fitted = True
        else:
            X_scaled = self.scaler.transform(X)
        
        # Conversión de etiquetas
        y_cat = to_categorical(y, num_classes=2)
        
        # División estratificada
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_cat, 
            test_size=test_size, 
            stratify=y,
            random_state=42
        )
        
        # Redimensionar para CNN
        return (
            X_train.reshape(*X_train.shape, 1),
            X_val.reshape(*X_val.shape, 1),
            y_train,
            y_val
        )

# %% ---------------------------
# 5. ARQUITECTURA DEL MODELO
# ------------------------------
class CancerDetector:
    def __init__(self):
        self.model = None
        self.tuner = None
        self.history = None
    
    def build_model(self, hp) -> tf.keras.Model:
        """
        Construye modelo con optimización bayesiana de hiperparámetros
        """
        model = models.Sequential([
            layers.Conv1D(
                filters=hp.Int('filters_1', 32, 128, step=32),
                kernel_size=hp.Int('kernel_size_1', 3, 10),
                activation='relu',
                input_shape=(1000, 1),
                kernel_regularizer=regularizers.l2(0.01)
            ),
            layers.MaxPooling1D(2),
            layers.BatchNormalization(),
            layers.Dropout(hp.Float('dropout_1', 0.2, 0.5, step=0.1)),
            
            layers.Conv1D(
                filters=hp.Int('filters_2', 64, 256, step=64),
                kernel_size=hp.Int('kernel_size_2', 3, 10),
                activation='relu',
                kernel_regularizer=regularizers.l2(0.01)
            ),
            layers.MaxPooling1D(2),
            layers.BatchNormalization(),
            layers.Dropout(hp.Float('dropout_2', 0.2, 0.5, step=0.1)),
            
            layers.Flatten(),
            layers.Dense(
                units=hp.Int('dense_units', 64, 256, step=64),
                activation='relu',
                kernel_regularizer=regularizers.l2(0.01)
            ),
            layers.Dropout(hp.Float('dropout_3', 0.2, 0.5, step=0.1)),
            layers.Dense(2, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=hp.Float('lr', 1e-4, 1e-2, sampling='log')),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        return model

    def tune_model(self, X_train, y_train, X_val, y_val, max_epochs: int = 100):
        """
        Optimización bayesiana de hiperparámetros
        """
        self.tuner = BayesianOptimization(
            self.build_model,
            objective='val_auc',
            max_trials=30,
            num_initial_points=10,
            directory='tuning',
            project_name='raman_cancer'
        )
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        self.tuner.search(
            X_train, y_train,
            epochs=max_epochs,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            batch_size=32,
            verbose=1
        )
        
        self.model = self.tuner.get_best_models(num_models=1)[0]
        logger.info("Mejor modelo seleccionado")

# %% ---------------------------
# 6. EVALUACIÓN Y EXPLICABILIDAD
# ------------------------------
class ModelEvaluator:
    @staticmethod
    def evaluate(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray):
        """
        Evaluación completa con métricas extendidas
        """
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Reporte de clasificación
        print(classification_report(y_true_classes, y_pred_classes))
        
        # Matriz de confusión
        plt.figure(figsize=(8,6))
        sns.heatmap(confusion_matrix(y_true_classes, y_pred_classes), 
                   annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Sano', 'Canceroso'],
                   yticklabels=['Sano', 'Canceroso'])
        plt.title('Matriz de Confusión')
        plt.show()
        
        # Curva ROC
        fpr, tpr, _ = roc_curve(y_true_classes, y_pred[:,1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curva ROC')
        plt.legend()
        plt.show()

    @staticmethod
    def explain_prediction(model: tf.keras.Model, sample: np.ndarray, background: np.ndarray):
        """
        Interpretación de predicciones usando SHAP
        """
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(sample.reshape(1, 1000, 1))
        
        plt.figure(figsize=(15, 4))
        plt.plot(sample.flatten(), label='Espectro Original')
        plt.plot(shap_values[0][0].sum(axis=0), label='Importancia SHAP', alpha=0.7)
        plt.title("Interpretación de la Predicción")
        plt.xlabel("Longitud de onda")
        plt.ylabel("Intensidad/Importancia")
        plt.legend()
        plt.show()

# %% ---------------------------
# 7. INTERFAZ GRÁFICA (GUI)
# ------------------------------
class RamanGUI:
    def __init__(self, model: tf.keras.Model, preprocessor: DataPreprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.root = tk.Tk()
        self.root.title("Analizador de Espectros Raman v1.0")
        self._setup_ui()
        
    def _setup_ui(self):
        """Configura los componentes de la interfaz"""
        # Marco principal
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        # Botón de carga
        self.load_btn = tk.Button(
            main_frame,
            text="Cargar Espectro CSV",
            command=self._load_file,
            width=25,
            height=2
        )
        self.load_btn.pack(pady=15)
        
        # Etiqueta de predicción
        self.pred_label = tk.Label(
            main_frame,
            text="Predicción: ",
            font=('Arial', 14, 'bold'),
            fg='#2c3e50'
        )
        self.pred_label.pack(pady=10)
        
        # Botón de explicación
        self.explain_btn = tk.Button(
            main_frame,
            text="Explicar Predicción",
            command=self._explain_prediction,
            state=tk.DISABLED,
            width=20,
            height=2
        )
        self.explain_btn.pack(pady=10)
        
        # Variables de estado
        self.current_sample = None
        
    def _load_file(self):
        """Maneja la carga de archivos"""
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV", "*.csv"), ("Text files", "*.txt")],
            title="Seleccionar archivo de espectro"
        )
        
        if file_path:
            try:
                # Validar tamaño y extensión
                if os.path.getsize(file_path) > 10 * 1024 * 1024:
                    messagebox.showerror("Error", "Archivo demasiado grande (>10MB)")
                    return
                
                if not file_path.lower().endswith(('.csv', '.txt')):
                    messagebox.showerror("Error", "Formato no soportado")
                    return
                
                # Cargar y preprocesar
                data = pd.read_csv(file_path, header=None).values
                if data.shape[1] != 1000:
                    messagebox.showerror("Error", "Formato incorrecto: Se requieren 1000 características")
                    return
                
                # Predecir
                scaled_data = self.preprocessor.scaler.transform(data)
                prediction = self.model.predict(scaled_data.reshape(-1, 1000, 1))
                result = "Canceroso" if np.argmax(prediction) == 1 else "Sano"
                confidence = np.max(prediction)
                
                # Actualizar UI
                self.pred_label.config(
                    text=f"Predicción: {result} (Confianza: {confidence:.2%})",
                    fg='#e74c3c' if result == "Canceroso" else '#2ecc71'
                )
                self.current_sample = scaled_data
                self.explain_btn.config(state=tk.NORMAL)
                
                # Mostrar espectro
                self._plot_spectrum(data[0])
                
            except Exception as e:
                messagebox.showerror("Error", f"Error procesando archivo:\n{str(e)}")
    
    def _plot_spectrum(self, spectrum: np.ndarray):
        """Visualiza el espectro cargado"""
        plt.figure(figsize=(10, 4))
        plt.plot(spectrum, color='#3498db')
        plt.title("Espectro Raman Cargado")
        plt.xlabel("Longitud de onda (nm)")
        plt.ylabel("Intensidad (u.a.)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def _explain_prediction(self):
        """Genera explicación de la predicción"""
        if self.current_sample is not None:
            background = self.preprocessor.scaler.transform(
                DataGenerator.generate_spectra(100)[0]
            ).reshape(100, 1000, 1)
            ModelEvaluator.explain_prediction(self.model, self.current_sample, background)
    
    def run(self):
        """Inicia la aplicación"""
        self.root.mainloop()

# %% ---------------------------
# 8. API REST
# ------------------------------
class RamanAPI:
    def __init__(self, model: tf.keras.Model, preprocessor: DataPreprocessor):
        self.app = Flask(__name__)
        self.model = model
        self.preprocessor = preprocessor
        self._setup_api()
        self._setup_swagger()
        
    def _setup_swagger(self):
        """Configura documentación Swagger/OpenAPI"""
        SWAGGER_URL = '/docs'
        API_URL = '/spec'
        
        swagger_ui = get_swaggerui_blueprint(
            SWAGGER_URL,
            API_URL,
            config={'app_name': "Raman Cancer Detector API"}
        )
        
        self.app.register_blueprint(swagger_ui, url_prefix=SWAGGER_URL)
        
        @self.app.route(API_URL)
        def spec():
            return jsonify(swagger(self.app))
    
    def _setup_api(self):
        """Configura los endpoints de la API"""
        @self.app.route('/predict', methods=['POST'])
        def predict():
            """
            Endpoint de predicción
            ---
            parameters:
              - name: data
                in: body
                required: true
                schema:
                  type: array
                  items:
                    type: number
                    example: 0.123
            responses:
              200:
                description: Resultado de la predicción
            """
            try:
                data = request.json.get('data')
                
                # Validación de entrada
                if not data or len(data) != 1000:
                    return jsonify({"error": "Se requieren exactamente 1000 valores"}), 400
                
                # Conversión a numpy
                arr_data = np.array(data, dtype=np.float 32).reshape(1, 1000, 1)
                prediction = self.model.predict(arr_data)
                result = "Canceroso" if np.argmax(prediction) == 1 else "Sano"
                confidence = np.max(prediction)
                
                return jsonify({
                    "prediction": result,
                    "confidence": confidence.tolist()
                }), 200
            
            except Exception as e:
                return jsonify({"error": str(e)}), 500

    def run(self):
        """Inicia la API"""
        self.app.run(host='0.0.0.0', port=5000)

# %% ---------------------------
# 9. MONITOREO DE RECURSOS
# ------------------------------
class ResourceMonitor(Thread):
    def run(self):
        """Monitorea el uso de recursos del sistema"""
        while True:
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            logger.info(f"Uso de recursos - CPU: {cpu}% | Memoria: {mem}%")
            time.sleep(60)

# %% ---------------------------
# 10. FUNCIÓN PRINCIPAL
# ------------------------------
if __name__ == "__main__":
    # Generar datos
    X, y = DataGenerator.generate_spectra(1000)
    
    # Preprocesar datos
    preprocessor = DataPreprocessor()
    X_train, X_val, y_train, y_val = preprocessor.preprocess(X, y)
    
    # Entrenar modelo
    detector = CancerDetector()
    detector.tune_model(X_train, y_train, X_val, y_val)
    
    # Evaluar modelo
    evaluator = ModelEvaluator()
    evaluator.evaluate(detector.model, X_val, y_val)
    
    # Iniciar GUI
    gui = RamanGUI(detector.model, preprocessor)
    Thread(target=gui.run).start()
    
    # Iniciar API
    api = RamanAPI(detector.model, preprocessor)
    Thread(target=api.run).start()
    
    # Iniciar monitoreo de recursos
    ResourceMonitor().start()