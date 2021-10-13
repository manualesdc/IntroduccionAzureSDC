#!/usr/bin/env python
# coding: utf-8

# # Seccion Entrenar y Registrar el modelo

# - Entrenamiento de un modelo en un Script
# 
# - Usar argumentos de script
# 
# - Registro de un modelo entrenado

# Nos conectamos utilizando el config.json

# In[2]:


import azure.core
from azureml.core import Workspace
ws = Workspace.from_config(path="config.json")


# # Entrenamiento de un modelo en un script
# Puede usar ScriptRunConfig para ejecutar un experimento basado en script que entrena un modelo de aprendizaje automático.
# 
# ### Escribir un script para entrenar un modelo
# Cuando utilice un experimento para entrenar un modelo, su secuencia de comandos debe guardar el modelo entrenado en la carpeta de resultados. Por ejemplo, el siguiente script entrena un modelo usando Scikit-Learn y lo guarda en la carpeta de resultados usando el paquete joblib:

# Creamos una carpeta **experiment_folder** de manera manual o mediante este codigo solo para guardar el archivo *.py* que vamos a generar

# In[23]:


import os
os.makedirs('training_folder', exist_ok=True)


# Generamos el script en un archivo .py

# In[24]:


get_ipython().run_cell_magic('writefile', 'training_folder/bank_training.py', 'from azureml.core import Run\nimport pandas as pd\nimport numpy as np\nimport joblib\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import accuracy_score\n\n# Obtenga el contexto de ejecución del experimento\nrun = Run.get_context()\n\n# Prepare el conjunto de datos \ndata = pd.read_csv(\'loan_bank/data_bank.csv\')\n\nX, y = data.iloc[:,:-1].values, data[\'def_pay_nxt_mont\'].values\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)\n\n# Entrenar el modelo de regresion logistica\nreg = 0.1\nmodel = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)\n\n# Calcular Accuracy\ny_predict = model.predict(X_test)\nacc = accuracy_score(y_test, y_predict)\nrun.log(\'Accuracy\', np.float(acc))\n\n# Guardar el modelo entrenado\nos.makedirs(\'outputs\', exist_ok=True)\njoblib.dump(value=model, filename=\'outputs/model.pkl\')\n\nrun.complete()')


# ## Ejecución del script como experimento
# 
# Para ejecutar la secuencia de comandos, cree un ScriptRunConfig que haga referencia a la carpeta y al archivo de secuencia de comandos. Por lo general, también debe definir un entorno Python (Conda) que incluya los paquetes requeridos por el script. En este ejemplo, el script usa Scikit-Learn, por lo que debe crear un entorno que lo incluya. El script también usa Azure Machine Learning para registrar métricas, por lo que debe recordar incluir el paquete azureml-defaults en el entorno.

# In[25]:


from azureml.core import Experiment, ScriptRunConfig, Environment
from azureml.core.conda_dependencies import CondaDependencies

# Crear un entorno para el experimento
sklearn_env = Environment("sklearn-env")

# Especificamos que paquetes requerira nuestro entorno
packages = CondaDependencies.create(conda_packages=['scikit-learn','pip','ipykernel'],
                                    pip_packages=['azureml-defaults'])
sklearn_env.python.conda_dependencies = packages

# Crear el script config
script_config = ScriptRunConfig(source_directory='training_folder',
                                script='bank_training.py',
                                environment=sklearn_env)

# Subir el experimento 
experiment = Experiment(workspace=ws, name='training-experiment')
run = experiment.submit(config=script_config)
run.wait_for_completion()


# # Usar argumentos de script
# Puede aumentar la flexibilidad de los experimentos basados en secuencias de comandos mediante el uso de argumentos para establecer variables en la secuencia de comandos.
# 
# ### Trabajar con argumentos de script
# Para usar parámetros en un script, debe usar una biblioteca como **argparse** para leer los argumentos pasados al script y asignarlos a variables. 
# 
# Por ejemplo, el siguiente script lee un argumento llamado **--reg-rate**, que se usa para establecer el hiperparámetro de la tasa de regularización para el algoritmo de regresión logística que se usa para entrenar un modelo.

# In[74]:


get_ipython().run_cell_magic('writefile', 'training_folder/bank_training2.py', '\nfrom azureml.core import Run\nimport argparse\nimport pandas as pd\nimport numpy as np\nimport joblib\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import accuracy_score\n\n# Obtenga el contexto de ejecución del experimento \nrun = Run.get_context()\n\n# Establecer hiperparametro de regularización\nparser = argparse.ArgumentParser()\nparser.add_argument(\'--reg-rate\', type=float, dest=\'reg_rate\', default=0.01)\nargs = parser.parse_args()\nreg = args.reg_rate\n\n# Preparar el dataset\ndata = pd.read_csv(\'loan_bank/data_bank.csv\')\n\nX, y = data.iloc[:,:-1].values, data[\'def_pay_nxt_mont\'].values\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)\n\n\n# Entrenamiento del LogisticRegression\nmodel = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)\n\n# Obtener el accuracy\ny_predict = model.predict(X_test)\nacc = accuracy_score(y_test, y_predict)\nrun.log(\'Accuracy\', np.float(acc))\n\n# Guardar el modelo entrenado\nos.makedirs(\'outputs\', exist_ok=True)\njoblib.dump(value=model, filename=\'outputs/model.pkl\')\n\nrun.complete()')


# ### Pasar argumentos a un script de experimento
# Para pasar valores de parámetros a una secuencia de comandos que se ejecuta en un experimento, debe proporcionar un valor de argumentos que contenga una lista de argumentos separados por comas y sus valores al **ScriptRunConfig**, como este

# In[75]:


# Crear un ScriptRunConfig
script_config = ScriptRunConfig(source_directory='training_folder',
                                script='bank_training2.py',
                                arguments = ['--reg-rate', 0.1],
                                environment=sklearn_env)


# # Registro de un modelo entrenado
# Después de ejecutar un experimento que entrena un modelo, puede usar una referencia al objeto **Run** para recuperar sus resultados, incluido el modelo entrenado.
# 
# ### Recuperar archivos de modelo
# Una vez que se ha completado la ejecución de un experimento, puede utilizar el método de ejecución de objetos **get_file_names** para enumerar los archivos generados. La práctica estándar es que los scripts que entrenan modelos los guarden en la carpeta de resultados de la ejecución.
# 
# También puede utilizar los métodos **download_file** y **download_files** del objeto de ejecución para descargar archivos de salida al sistema de archivos local.

# In[28]:


# "run" es una referencia a la ejecución de un experimento completado 

# Lista los archivos generados por el experimento
for file in run.get_file_names():
    print(file)

# Descargar un archivo con nombre
run.download_file(name='outputs/model.pkl', output_file_path='model.pkl')


# In[29]:


metrics = run.get_metrics()
for key in metrics.keys():
        print(key, metrics.get(key))


# ### Registro de un modelo
# El registro de modelos le permite realizar un seguimiento de varias versiones de un modelo y recuperar modelos para realizar inferencias (predecir valores de etiqueta a partir de datos nuevos). Cuando registra un modelo, puede especificar un nombre, descripción, etiquetas, marco (como Scikit-Learn o PyTorch), versión del marco, propiedades personalizadas y otros metadatos útiles. 
# 
# El registro de un modelo con el mismo nombre que un modelo existente crea automáticamente una nueva versión del modelo, comenzando con 1 y aumentando en unidades de 1.
# 
# 

# Si tenemos una referencia al objeto **run** utilizada para entrenar el modelo, puede usar su método **register_model** como se muestra aquí:

# In[30]:


from azureml.core import Model

run.register_model( model_name='classification_model',
                    model_path='outputs/model.pkl', # run outputs path
                    description='Un modelo de clasificacion',
                    tags={'Training context': 'Script Parametrizado','RelacionExperimento':'Si'},
                    model_framework=Model.Framework.SCIKITLEARN,
                    model_framework_version='0.20.3')


# Alternativamente, para registrar un modelo desde un archivo local sin que este relacionado a un **Experimento**, puede usar el método de registro del objeto **Model** como se muestra aquí:

# In[31]:


from azureml.core import Model

model = Model.register(workspace=ws,
                       model_name='classification_model_credit2',
                       model_path='model.pkl', # local path
                       description='Un modelo de clasificacion 2',
                       tags={'Training context': 'Script Parametrizado','RelacionExperimento':'No'},
                       model_framework=Model.Framework.SCIKITLEARN,
                       model_framework_version='0.20.3')


# ### Visualización de modelos registrados
# Puede ver los modelos registrados en Azure Machine Learning Studio. También puede usar el objeto Modelo para recuperar detalles de modelos registrados como este:

# In[32]:


from azureml.core import Model

for model in Model.list(ws):
    # Obtener el nombre del modelo y la versión generada automáticamente 
    print(model.name, 'version:', model.version)


# ## Registra una nueva versión del modelo
# 
# Ahora que ha entrenado un nuevo modelo, puede registrarlo como una nueva versión en el espacio de trabajo.

# Como se explico antes cuando registramos el modelo con el mismo nombre por el parametro **model_name** entonces funcionara el control de versiones.Para no estar creando otra ejecución registramos el mismo modelo para ver que se agrego con el mismo nombre el modelo pero con otra version

# In[77]:


from azureml.core import Model

# Registrar el modelo
run.register_model(
    model_path='outputs/model.pkl', 
    model_name='classification_model',
    tags={'Training context': 'Script Parametrizado','RelacionExperimento':'Si'},
    properties={'Accuracy': run.get_metrics()['Accuracy']}
)


# In[42]:



# Listar los modelos registrados
for model in Model.list(ws):
    print(model.name, 'version:', model.version)
    for tag_name in model.tags:
        tag = model.tags[tag_name]
        print ('\t',tag_name, ':', tag)
    for prop_name in model.properties:
        prop = model.properties[prop_name]
        print ('\t',prop_name, ':', prop)
    print('\n')


# También puede ver los modelos registrados en su área de trabajo en la página **Models**
