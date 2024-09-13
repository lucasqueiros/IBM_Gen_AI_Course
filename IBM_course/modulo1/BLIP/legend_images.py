from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

#Inicializando o processador e o modelo do hugging face
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

#Carregando uma imagem
image = Image.open("/home/lucasqueiros/Documentos/CC/IBM_course/Images/Lamborghini.jpg")

#Preparando a imagem
inputs = processor(image, return_tensors="pt")

#Geração de legendas
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0], skip_special_tokens=True)

print("Legenda gerada:", caption)
