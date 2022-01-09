# IA-Clasificatoria

#----------------------------------------------------------------#

Proyecto de redes neuronales.
Realizado por Jorge Armas Morales y Marcos Ismael Medina Castellano
Proyecto realizado para la asignatura de Fundamentos de los Sistemas Inteligentes
Grado en Ingeniería Informática

#----------PLANTEAMIENTO-------------#

La idea de esta red neuronal es crear un clasificador de imágenes que pueda clasificar diferentes paisajes urbanos
bajo 6 clases diferentes. Para todo el proyecto se utiliza la librería keras.
A la hora de configurar el set de entrenamiento de utiliza la técnica de data augmentation.
La estructura de la red consta de 4 capas convolutivas de 16, 32, 64 y 128 neuronas respectivamente con un kernel 6x6.
Las capas se intercalan siempre con una capa de maxpooling 2x2 y otra de dropout de 25% (salvo por un 50% en la capa final)
con tal de evitar el maxpooling.
Para concluir la estructura tenemos un proceso de flatten, una capa fully conected con las mismas condiciones anteriores y 128 neuronas
y la última capa fully connected con función de activación softmax para obtener la clase de la imagen evaluada.

Con esta estructura se consigue una precisión de validación del 86.7%

#--------REFERENCIAS---------#

GPU utilizada: GTX 1660 Ti Max Q
Dataset utilizado: https://www.kaggle.com/puneet6060/intel-image-classification
