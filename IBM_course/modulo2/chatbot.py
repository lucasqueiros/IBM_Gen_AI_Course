# importando biblioteca transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# nesse exemplo usaremos o modelo do facebook por ser open-source e eficiente
model_name = "facebook/blenderbot-400M-distill"

# inicializando o modelo e o tokenizador
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# criando uma lista para historico de conversa
conversation_history = []

#adicionando tudo dentro de um laço
while True:
    history_string = "\n".join(conversation_history)

    # exemplo de entrada inicial (tentei em portugues mas a saída gerada foi muito ruim)
    input_text = input("> ")

    #tokenizando o historico de conversa e a entrada inicial
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

    # gerando saida
    outputs = model.generate(**inputs)

    #decodificando a saida
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    #printando a saida
    print(response)
    
    #adicionando input e output ao historico da conversa
    conversation_history.append(input_text)
    conversation_history.append(response)
    

