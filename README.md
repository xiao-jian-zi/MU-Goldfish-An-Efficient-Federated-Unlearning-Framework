# MU-Goldfish-An-Efficient-Federated-Unlearning-Framework
PyTorch code for an efficient federated unlearning approach described in the paper "Goldfish: An Efficient Federated Unlearning Framework".
## 1 Requirements
We recommended the following dependencies.

* python==3.8
* pytorch==1.11.0
* torchvision==0.12.0
* numpy==1.22.4

For more recommended dependencies, please refer to the file [`requirements.txt`](https://github.com/xiao-jian-zi/MU-Goldfish-An-Efficient-Federated-Unlearning-Framework/blob/main/requirements.txt).

## 2 How to use
### 2.1 Training new models

Run `train_[dataset_name].py` to obtain the trained models: 

```bash
python train_[dataset_name].py
```
 You can specify the number of training epochs by setting the `epoch_num` in the file. If necessary, please modify the path to the dataset; otherwise, the code will automatically download the dataset to the default path when executed.


### 2.2 Unlearning
Run `unlearning_[dataset_name].py` to perform the unlearning process: 
```bash
python unlearning_[dataset_name].py
```
To customize the unlearning process, please make the following adjustments in the file:
* Modify the `epoch_num` value to specify the number of training epochs.
* Modify the `teacher_net_path` to set the save path for the teacher model.
* Modify the `forget_Proportion` to define the size of the forgetting dataset.
  
These configurations will allow you to tailor the unlearning process to your specific requirements. Ensure that the values are correctly set before running the training script to avoid any errors or unexpected results.

### 2.3 Federated learning
Run the `Fed_mnist_even.py` file to perform federated learning under the condition of even local data distribution.

```bash
python Fed_mnist_even.py
```
Run the `Fed_mnist_uneven.py` file to perform federated learning under the condition of uneven local data distribution.

```bash
python Fed_mnist_uneven.py
```
Run the `Fed_mnist_unlearning.py` file to perform federated unlearning.
```bash
python Fed_mnist_unlearning.py
```
## 3 Code Reference
For detailed code explanations and best practices, please refer to
* [https://github.com/akamaster/pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10)
* [https://github.com/IMoonKeyBoy/The-Right-to-be-Forgotten-in-Federated-Learning-An-Efficient-Realization-with-Rapid-Retraining](https://github.com/IMoonKeyBoy/The-Right-to-be-Forgotten-in-Federated-Learning-An-Efficient-Realization-with-Rapid-Retraining)
* [https://github.com/vikram2000b/bad-teaching-unlearning](https://github.com/vikram2000b/bad-teaching-unlearning)
