# ActionVerification
 
## Background
This is the *internal beta* implementation of our Action-Verificaiton project.

## Install
`$ TBD`

Our environment:
> Ubuntu 18.04.5 LTS
>
> CUDA 10.2.89
>
> Pytorch 1.7.1

## Usage

The detailed setup of model, training, dataset are configured in [config file], dir *configs/* contains some examples. The following are main commands used to play with this project.

For training:

`$ CUDA_VISIBLE_DEVICES=[X,X,X,X] python train.py --config [config file]`

For testing:

*Mode 1*: Evaluate all models in [root_path/saved_models] and save log file to [root_path/logs].

`$ CUDA_VISIBLE_DEVICES=[X,X,X,X] python eval.py --config [config file] --root_path [root_path]`

*Mode 2*: Evaluate one model in [model_path] and save log file to [temp_log] if not especially speicify.

`$ CUDA_VISIBLE_DEVICES=[X,X,X,X] python eval.py --config [config file] --model_path [model_path]`



