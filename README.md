# GII

## Training and Evaluation
The experiments will load the dataset from the `data/` directory, then split it into training, validation and test sets in customized ratio. The default ratio (also the ratio we use in the paper) is `0.75(training): 0.25(testing)`. To run the experiments in the paper, use this command:

```train
python run_exp_opennet.py -n <network_type> cnn -ds <dataset>[mnist, cifar10, msadjmat, android] -m <loss_function_type> ii -trc_file <known_classes_file>[mnist_trc_list, msadjmat_trc_list, android_trc_list] -o <output_file> 
```
e.g. 
```
# MNIST dataset
python run_exp_opennet.py -n cnn -ds mnist -m ii  -trc_file "data/mnist_trc_list" -o "data/results/cnn/mnist" 
```
