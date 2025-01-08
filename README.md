# PS Unlearning

The code for paper "Prototype Surgery: Modifying Neural Prototypes via Soft Labels for Enhanced Machine Unlearning Efficiency".

## About the project

Prototype Surgery primarily focuses on the prototype level to facilitate machine unlearning. Before performing unlearning, it is essential to prepare the original model along with its corresponding dataset.

In our study, we evaluate the approach on four popular datasets—MNIST, CIFAR-10, CIFAR-100, and TinyImageNet—using LeNet5, ResNet20, ResNet50, and VGG16 models, respectively.

## Getting started

File structure

```os
PS-Unlearning-Master
├── utils
│   ├── data.py
│   ├── models.py
│   └── seed.py
└── main.py
```

There are several parts of the code:
- `data.py`: This file contains code to process and load data.
- `models.py`: This file contains operations related to models.
- `seed.py`: This file is used to seed everything.
- `main.py`: The main code for Naive PS and PS.


### Requirements

To set up the required environment for this project, please ensure you have Python (>=3.8) installed. All necessary dependencies are listed in the `requirements.txt` file. You can easily install them by running:

```bash
pip install -r requirements.txt
```

This command will automatically install all the required packages and their compatible versions for the project.

### Hyper Parameters

| Parameter       | Default Value    | Description                                                                                     |
|------------------|------------------|-------------------------------------------------------------------------------------------------|
| `model_type`    | `"MNIST"`        | The type of dataset and model to use. Options: `MNIST`, `CIFAR10`, `CIFAR100`, `TINYIMAGENET`.  |
| `expid`         | `0`              | Experiment ID to specify which pre-trained model to load. Valid range: `0` to `9`.             |
| `ratio`         | `0.1`            | The proportion of data to be unlearned in the original dataset.                                |
| `seed`          | `209`            | The random seed for ensuring reproducibility across experiments.                               |
| `EPOCH`         | Varies by dataset| The total number of epochs for fine-tuning. |
| `LR`            | Varies by dataset| Learning rate for fine-tuning. |
| `LEN`           | Varies by dataset| The number of samples per class for the unlearn/remain datasets: `6000` (MNIST), `5000` (CIFAR10), `500` (CIFAR100, TinyImageNet). |

Hyperparameters related to fine-tuning can be modified directly in `main.py`.

#### Notes
- `model_type`, `expid`, and `ratio` are set via command-line arguments.
- `EPOCH`, `LR`, and `LEN` are determined based on the selected `model_type`.

### Run
You could run `main.py` in your python IDE directly.
The example codes below show the workflow to perform a complete fingerprinting process, which is in `main.py`.

```python
def main(args):
    # Set random seed.
    seed.seed_everything(args.seed)
    # Naive PS unlearning.
    NaivePS(args.model_type, args.expid, args.ratio)
    # PS unlearning.
    PS(args.model_type, args.expid, args.ratio)
```

You can also run main.py using the cmd command.
```python
$ python main.py --model_type "MNIST" --expid 0 --ratio 0.1 --seed 209
```


<br>

#### Note
- A GPU is not required, but we recommend using one to increase the running speed. 

<br>
