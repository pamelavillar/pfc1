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

        layers.append(nn.Linear(prev_size,output_size)) 

        self.network = nn.Sequential(*layers)

    def forward(self,x):
        return self.network(x)
    

class InjectionBlock(nn.Module):
    name = "generic"

    def __init__(self,):
        super(InjectionBlock,self).__init__()
        self.counter = 0
        self.generating = False 

    def restart_generator_counter(self): 
        self.counter = 0
        self.generating = True

    def initialize_modules(self, modules = None): 
        if modules is None:
            modules = self.modules() 
        for module in modules:
            if isinstance(module,nn.Linear):
                for name, param in module.named_parameters(): 
                    if "weight" in name:
                       
                        nn.init.kaiming_uniform_(param, a = math.sqrt(5))
                    elif "bias" in name:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight) 
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0 
                        nn.init.uniform_(param, -bound,bound) 
        



class InputInjectionBlock(InjectionBlock):
    name = "input_injection_block"

    def __init__(self,hidden_dim:int = 256, model_dim:int = 512, num_covariate:int = 0, num_layers: int = 1):
        super(InputInjectionBlock,self).__init__()

        self.hidden_dim = hidden_dim 
        self.model_dim = model_dim 
        self.num_covariates = num_covariate
        self.num_layeres = num_layers 

        self.cov_in = nn.Linear(self.num_covariates, self.hidden_dim)
        self.concat_dim = self.hidden_dim*2
        self.concat_layer = FeedForwardNN(self.concat_dim, [self.hidden_dim] * self.num_layeres, self.model_dim)
        self.emb_in = nn.Linear(self.model_dim, self.hidden_dim)

    def forward(self, input_embeds, past_covariates, is_decoder =False):
        x = self.emb_in(input_embeds)
        if self.generating and is_decoder:
            past_covariates = past_covariates[: ,self.counter, :].unsqueeze(1)
            self.counter +=1
        
        x_cov  = self.cov_in(past_covariates) 

        x = torch.cat([x,x_cov], axis =- 1) 

        x = nn.ReLU()(x) 

        #residual connection
        return input_embeds + self.concat_layer(x)


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
        self.concat_layer = FeedForwardNN(self.concat_dim, [self.hidden_dim] * self.num_layeres, self.vocab_size) 

        self.hidden_state_out = nn.Linear(self.model_dim, self.hidden_dim)
        self.loss_fct = CrossEntropyLoss(ignore_index=-100)

    def compute_loss(self, logits,labels):
        labels = labels.to(logits.device) 
       
        loss = self.loss_fct(logits.view(-1,logits.size(-1)),labels.view(-1))
        return loss
    

    
    def forward(self, logits, labels, future_covariates, last_hidden_state):
        x = self.hidden_state_out(last_hidden_state) #
        
        if self.generating: 
            x_cov = future_covariates[: ,self.counter, :].unsqueeze(1)
        else:
            x_cov = future_covariates

        if x_cov.flatten().isnan().sum() > 0: 
            print(1)
        
        x_cov = self.cov_out(x_cov) 

        x = torch.concatenate([x,x_cov], axis =- 1)  

        x = nn.ReLU()(x) 
        
        logits =  self.concat_layer(x) + logits

        if self.training:
            loss = self.compute_loss(logits,labels) 
        else:
            loss = None
            self.counter+=1 

        return logits,loss 