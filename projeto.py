# Para capturar os quadros
import cv2

# Para processar o array de imagens
import numpy as np


# importe os módulos tensorflow e carregue o modelo
import tensorflow as tf

model = tf.keras.models.load_model("keras_model.h5")

# Anexando a câmera indexada como 0, com o software da aplicação
camera = cv2.VideoCapture(0)

# Loop infinito
while True:

	# Lendo / requisitando um quadro da câmera 
	status , frame = camera.read()

	# Se tivemos sucesso ao ler o quadro
	if status:

		# Inverta o quadro
		frame = cv2.flip(frame , 1)
  
		resizedframe = cv2.resize(frame,(224,224))
  
		resizedframe = np.expand_dims(resizedframe,axis=0)
  
		resizedframe = resizedframe/255
  
		predictions = model.predict(resizedframe)	
  
		print(f"pedra:{predictions[0][0]},papel:{predictions[0][1]},tesoura:{predictions[0][2]}")

		# Exibindo os quadros capturados
		cv2.imshow('feed' , frame)

		# Aguardando 1ms
		code = cv2.waitKey(1)
		
		# Se a barra de espaço foi pressionada, interrompa o loop
		if code == 32:
			break

# Libere a câmera do software da aplicação
camera.release()

# Feche a janela aberta
cv2.destroyAllWindows()
