# A domain-label-guided translation model for molecular optimization

# Installation:
1. Install conda / minicaonda
2. From the main folder run:\
    i. conda env create -f environment.yml\
    ii. conda activate DLTM



# Training:
From the main folder run:

1. python main.py 
main.py is the main training file, contains most of the hyper-parameter and configuration setting.
The most of them are listed below:
```
    data_path : the path of the datasets
    log_dir : where to save the logging info
    save_path = where to save the checkpoing
    max_length : the maximun molecular length 
    embed_size : the dimention of the token embedding
    c_model : the dimention of Transformer c_model parameter
    head : the number of Transformer attention head
    num_layers : the number of Transformer layer
    batch_size : the training batch size
    lr_T : the learning rate for the Transformer
    lr_C : the learning rate for the Classfier
    iter_T : the number of the Classfier update step pre training interation
    iter_C : the number of the Transformer update step pre training interation
    dropout : the dropout factor for the whole model
    eval_steps : the number of steps to evaluate model info
    slf_factor : the weight factor for the self reconstruction loss
    cyc_factor : the weight factor for the cycle reconstruction loss
    map_factor : the weight factor for the cycle mapping loss
    cls_factor : the weight factor for the molecular domain label loss
```
You can adjust them in the Config class from the 'main.py' for diffrent molecular tasks.



## Outputs

Update: You can find the outputs of our model in the "outputs" folder.