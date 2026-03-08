# BRAIN Moedl
***

## Requirements
```commandline
cond env create -f environment.yaml
conda activate BRAIN_env
```

## Codes

### Specify your configuration file
Modify the configuration file based on our templates in <font color=violet><b>configs/Template-BRAIN.yaml</b></font>  
Don't forget to specify your checkpoint path and dataset path.
### Specity your shell

If you wish to sample the example dataset 
```commandline
python3 test.py --config configs/Template-BRAIN.yaml --gpu_ids 0 --resume_model pth/to/model_pth
```

Note that optimizer checkpoint is not needed and specifying checkpoint path in commandline has higher priority than specifying in configuration file.

## Checkpoints
The model checkpoints after training on our NCCT-DWI dataset are provided.The model file is available for download via Baidu Netdisk:
Link: https://pan.baidu.com/s/1FlXQJSg7s4im_EMGBH_4lg?pwd=re8i


