from torch import nn
from torch.nn import CrossEntropyLoss
import torch 

import math

class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FeedForwardNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_sizes
        self.output_size = output_size

        layers = []
        prev_size = input_size

        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size,size))
            layers.append(nn.ReLU())
            prev_size = size

        layers.append(nn.Linear(prev_size,output_size)) # weight  = prev_size x output_size , bias = prev_size

        self.network = nn.Sequential(*layers)

    def forward(self,x):
        return self.network(x)
    

#una capa linear hace : y = Wx + b , donde W es weight y b bias

#clase para bloques de una red neuronal, es una clase par que otros hereden
#componente de pytorch
class InjectionBlock(nn.Module):
    name = "generic"

    def __init__(self,):
        super(InjectionBlock,self).__init__()
        self.counter = 0
        self.generating = False #nos indica si el bloque esta en algun proceso de generacopm

    def restart_generator_counter(self): #voy a comenzar un nuevo proceso de generacion
        self.counter = 0
        self.generating = True

    #su objetivo es inicializar los parametros de las capas linear
    def initialize_modules(self, modules = None): 
        if modules is None:
            modules = self.modules() #devuelve los modules que existen en ese objeto
            #que modulos?: linear,relu,linear
        for module in modules:
            #esto es para comenzar el entrenamiento con pesos y biases en escalas razonables para que las señales  y gradientesno exploten  desde el inicio
            if isinstance(module,nn.Linear): #busca solamente las capas lienar 
                for name, param in module.named_parameters(): # para cada modulo linear se tiene un weigth y bias
                    if "weight" in name:
                        #en vez de dejar los pesos con valores arbitrarios, se generan valores adecuados para comenzar el entrenamiento,
                        # es muy comun cuando se tiene activaciones RELU 
                        nn.init.kaiming_uniform_(param, a = math.sqrt(5))
                    elif "bias" in name:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight) # igual a input size
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0 #calcula el limite
                        nn.init.uniform_(param, -bound,bound) #inicializa los bisese aleatoriamente entre los limitesj
        



class InputInjectionBlock(InjectionBlock):
    name = "input_injection_block"

    def __init__(self,hidden_dim:int = 256, model_dim:int = 512, num_covariate:int = 0, num_layers: int = 1):
        super(InputInjectionBlock,self).__init__()

        self.hidden_dim = hidden_dim #Es una dimensión intermedia utilizada para procesar tanto el embedding como las covariables.
        self.model_dim = model_dim #dimencion de los embeddings que utiliza el modelo principal
        self.num_covariates = num_covariate
        self.num_layeres = num_layers #capas ocultas del FFN

        self.cov_in = nn.Linear(self.num_covariates, self.hidden_dim)
        self.concat_dim = self.hidden_dim*2
        self.concat_layer = FeedForwardNN(self.concat_dim, [self.hidden_dim] * self.num_layeres, self.model_dim) # en medio es una lista de dos hidden_dim

        self.emb_in = nn.Linear(self.model_dim, self.hidden_dim) #genera una capa para trasnformar  los embeddings para que sean hidden_dim

    def forward(self, input_embeds, past_covariates, is_decoder =False):
        x = self.emb_in(input_embeds) # = hemb(zt−1) W (emb)
        if self.generating and is_decoder: #modo autoregresivo/generativo, procesa las covariables paso por paso
            past_covariates = past_covariates[: ,self.counter, :].unsqueeze(1)
            self.counter +=1
        
        x_cov  = self.cov_in(past_covariates) # = x(t−1) W (cov)

        x = torch.cat([x,x_cov], axis =- 1) 

        x = nn.ReLU()(x) 

        #residual connection
        return input_embeds + self.concat_layer(x) # = FFN(RELU(x_cov, x))


class OutputInjectionBlock(InjectionBlock):
    name = "output_injection_block"

    def __init__(self,hidden_dim:int = 256, model_dim:int = 512, num_covariate:int = 0, num_layers: int = 1, vocab_size: int = 4096):
        super(OutputInjectionBlock,self).__init__()

        self.hidden_dim = hidden_dim 
        self.model_dim = model_dim
        self.num_covariates = num_covariate
        self.num_layeres = num_layers #capas ocultas del FFN
        self.vocab_size = vocab_size

        self.cov_out = nn.Linear(self.num_covariates, self.hidden_dim)
        self.concat_dim = self.hidden_dim*2
        self.concat_layer = FeedForwardNN(self.concat_dim, [self.hidden_dim] * self.num_layeres, self.vocab_size) # en medio es una lista de dos hidden_dim

        self.hidden_state_out = nn.Linear(self.model_dim, self.hidden_dim)
        self.loss_fct = CrossEntropyLoss(ignore_index=-100) #si alguna posicion de lables contiene -100, no calcules perdida para esa posicion, por que? porque puede haber posiciones que no queremos utilizar para entrenar

    def compute_loss(self, logits,labels): #NO ENTIENDO NADA
        labels = labels.to(logits.device) #mueve los logits al mismo dispositivo que logits, osea si estan en CPU , cambiar a GPU
        #Para cada posición de la secuencia, tengo 4096 posibles respuestas. Compara esas predicciones con el token correcto de labels y calcula qué tan equivocadas fueron.
        #logits: [20, 4096]
        # labels: [20]
        loss = self.loss_fct(logits.view(-1,logits.size(-1)),labels.view(-1))
        return loss
    

    #logits: prediccion orignal del modelo, last_hidden_state:represetnacion interna producida por el modelo
    #labels = respuesta correcta, usada para calcular la perdida
    def forward(self, logits, labels, future_covariates, last_hidden_state):
        x = self.hidden_state_out(last_hidden_state) #
        
        # como que si esta generando, osea generando que?
        if self.generating: #modo autoregresivo/generativo, procesa las covariables paso por paso
            x_cov = future_covariates[: ,self.counter, :].unsqueeze(1)
        #durante entrenamiento... usa todas las variables futuras de una vez
        else:
            x_cov = future_covariates

        if x_cov.flatten().isnan().sum() > 0: # esto es solo para veriricar que x_cov tienen algun NAN, solo es debugging
            print(1)
        
        x_cov = self.cov_out(x_cov) # = x(t−1) W (cov)

        x = torch.concatenate([x,x_cov], axis =- 1)  #CUAL ES LA DIFERENCIA CON CAT Y CONCATENATE, ningunaa solo sintaxis

        x = nn.ReLU()(x) 
        
        logits =  self.concat_layer(x) + logits

        if self.training:
            loss = self.compute_loss(logits,labels) #hace backpropagation para ir actualizando los pesos
        else:
            loss = None
            self.counter+=1 #siguiente covariable

        return logits,loss 