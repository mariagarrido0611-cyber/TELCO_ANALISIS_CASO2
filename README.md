# Telco Churn Predictor

Aplicación Streamlit para estimar la probabilidad de fuga (churn) de un cliente de telecomunicaciones a partir de un modelo de Machine Learning entrenado en `TELCO_FINAL.ipynb`.

## Estructura del repositorio

```text
telco-churn/
├── app.py
├── modelo_churn.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

> **Importante:** `modelo_churn.pkl` no viene incluido en este paquete. Debes generarlo desde el notebook y agregarlo al repositorio antes de desplegar. La aplicación no vuelve a entrenar el modelo: carga el pipeline serializado.

## 1. Generar el modelo

Abre `TELCO_FINAL.ipynb` en el entorno donde ejecutaste el entrenamiento y corre las celdas hasta la sección **Despliegue**. La celda de exportación debe ejecutar:

```python
import joblib

joblib.dump(modelo_final, "modelo_churn.pkl")
```

Descarga `modelo_churn.pkl` y cópialo en la misma carpeta que `app.py`.

El modelo guardado en el notebook es `best_gb`, un pipeline de `imbalanced-learn` con preprocesamiento, SMOTE y Gradient Boosting. Por eso `imbalanced-learn` y `scikit-learn` deben estar instalados en el entorno de despliegue.

## 2. Probar localmente

Recomendado: usa la misma versión de Python y las mismas versiones de `scikit-learn` e `imbalanced-learn` que utilizaste para entrenar y guardar el modelo. Los archivos pickle/joblib pueden fallar si se cargan con versiones incompatibles.

En Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Abre `http://localhost:8501`.

## 3. Subir a GitHub

1. Crea un repositorio en GitHub, por ejemplo `telco-churn-streamlit`.
2. Sube `app.py`, `requirements.txt`, `README.md`, `.gitignore` y `modelo_churn.pkl`.
3. Confirma que `modelo_churn.pkl` aparece en el repositorio y que no quedó dentro de una subcarpeta distinta.

Con Git instalado, desde esta carpeta puedes ejecutar:

```bash
git init
git add app.py requirements.txt README.md .gitignore modelo_churn.pkl
git commit -m "Deploy Telco Churn Streamlit app"
git branch -M main
git remote add origin https://github.com/USUARIO/telco-churn-streamlit.git
git push -u origin main
```

Reemplaza `USUARIO` por tu usuario u organización de GitHub. Si el archivo del modelo supera el límite de carga de GitHub, utiliza Git LFS.

## 4. Desplegar en Streamlit Community Cloud

1. Entra a https://share.streamlit.io/ e inicia sesión con GitHub.
2. Selecciona **Create app**.
3. Elige el repositorio `telco-churn-streamlit`.
4. Selecciona la rama `main`.
5. En **Main file path**, escribe `app.py`.
6. En **Advanced settings**, selecciona Python 3.11 si ese fue el entorno usado para entrenar el modelo.
7. Pulsa **Deploy** y revisa los logs si aparece un error.

## 5. Problemas frecuentes

- **`FileNotFoundError: modelo_churn.pkl`:** falta el archivo, tiene otro nombre o no está en la raíz del repositorio.
- **`ModuleNotFoundError`:** falta una dependencia en `requirements.txt`.
- **Error al deserializar el modelo / `InconsistentVersionWarning`:** instala las mismas versiones de `scikit-learn`, `imbalanced-learn` y Python usadas al entrenar. Consulta las versiones impresas por la celda de versiones del notebook y fija esas versiones en `requirements.txt`.
- **La app abre, pero falla al calcular:** revisa que el modelo haya sido entrenado con las mismas columnas y categorías que envía `app.py`.

## Nota sobre el alcance

La interfaz solicita algunas variables al usuario y asigna valores predeterminados al resto de las variables predictoras. Por tanto, el resultado es una estimación condicionada a esos valores predeterminados; no equivale a capturar el perfil completo del cliente.
