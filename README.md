# enid - Deep Learning for Medical Claims

## Summary

**enid** is a Python package for training and deploying adverse event prediction deep learning model based on medical claims supported by [Inovalon Inc.](https://www.inovalon.com/) The package includes the entire pipeline for vectorizing patient claim codes, claim code embedding training, predictive model training and deployment, as well as interpretation of the model. The models leverage wide range of medical claims, including all diagnoses and procedures as International Classification of Diseases (ICD) version 9/10 diagnosis/procedures, Current Procedural Terminology (CPT) and Healthcare Common Procedure Coding System (HCPCS); also medications as National Drug Code (NDC); as well as lab tests and results as Logical Observation Identifiers Names and Codes (LOINC).

**enid** provides stable Python APIs based on [Tensorflow](https://www.tensorflow.org/) enabling both CPU and GPU usage. Keep up to date with version releases and issue reporting by
emailing to
[ywang2@inovalon.com](ywang2@inovalon.com), [mmcclellan@inovalon.com](mmcclellan@inovalon.com) or [wkinsman@inovalon.com](wkinsman@inovalon.com).

## Installation

To download built wheel files:

| Build Type      | Status | Artifacts |
| ---             | ---    | ---       |
| **Linux**   | [![Status](imgs/linux-build.svg)]() | [Py3 wheel](https://www.inovalon.com/) |
| **Windows** | [![Status](imgs/win-build.svg)]() | [Py3 wheel](https://www.inovalon.com/) |

To install on Linux machine:

```
$ pip install enid-1.0.0-py3-manylinux1_x86_64.whl
```

To install on Windows machine:

```
$ pip install enid-1.0.0-py3-win_amd64.whl
```

Or install from source

```
$ python setup.py install
```

## API Documentation

### 1. data_helper

This module contains the vectorizer prior feeding data to any models.

#### class HierarchicalVectorizer

```python
enid.data_helper.HierarchicalVectorizer()
```

```HierarchicalVectorizer``` aims to vectorize claim data into two-level event containers for further use. Specifically, output the time deltas of all encounters and index of each claim codes according to our fixed indexing dictionary. Once initialized, it contains \_\_call\_\_ function to take a dictionary (JSON) with patient's encounter date and codes as input, and return two numpy arrays:

##### Parameters
```seq``` : JSON (dict) type object with format "YYYY-mm-dd": ["code_type-code"]. e.g.
```json
{
    "2015-01-05": [
        "CPT-97112",
        "ICD9DX-7813",
        "CPT-97530",
        "POS-11"
    ],
    ...
    "2016-10-11": [
        "REVENUE-0510",
        "ICD10DX-Z23",
        "CPT-90460"
    ]
}
```

```max_sequence_length``` : int, the fixed padding number of encounters. If the number of encounters is larger than max\_sequence\_length, only use the latest max\_sequence\_length; if the number of encounters is smaller than max\_sequence\_length, fill the earlier sequence with null index.

```max_token_length``` : int, the fixed padding number within each encounter. If the number of claim codes is larger than max\_token\_length, randomly sample max\_token\_length, number of codes; if the number of encounters is smaller than max\_token\_length, fill the later sequence with null index.
    
##### Return

```T``` : numpy array, shape (```max_sequence_length```,). Standardized time deltas between each two encounters

```X```: numpy array, shape (```max_sequence_length```, ```max_token_length```). The index of each claim codes based on dictionary

##### Examples
```python
>>> from enid.vectorizer import HierarchicalVectorizer
>>> vec = HierarchicalVectorizer()
>>> seq = {"2015-01-05": ["CPT-97112", "ICD9DX-7813",  "CPT-97530",  "POS-11"],
           ...
           "2016-10-11": ["REVENUE-0510", "ICD10DX-Z23", "CPT-90460" ]}
>>> vec(seq, 30, 20)[0]
array([17,  1,  2,  1,  1,  1,  1,  1,  1,  1,  3,  2,  1,  1,  1,  3,  3,
        1,  3,  1,  1,  5,  1,  2,  4,  7,  1,  8,  2,  3]

>>> vec(seq, 30, 20)[1]
array([[  423,    50, 64070, 64070, 64070, 64070, 64070, 64070, 64070,
        64070, 64070, 64070, 64070, 64070, 64070, 64070, 64070, 64070,
        64070, 64070],
       ...
       [ 2829,  2214,   546,     4,     0,   329,   142, 64070, 64070,
        64070, 64070, 64070, 64070, 64070, 64070, 64070, 64070, 64070,
        64070, 64070]])
```

### 2. claim2vec

This module contains the training module of medical claim embedding, mimics Word2Vec skipgram. Unlike Word2Vec assuming nearest token are related, due to claims are only attached with date but no timestamp, the model assume all the claims within a day are related.

#### class Claim2Vec

```python
enid.claim2vec.Claim2Vec(dictionary, batch_size=128, embedding_size=256, learning_rate=0.1, decay_steps=50000,
                         decay_rate=0.95, num_sampled=64, valid_size=16)
```

##### Parameters
* ```dictionary``` : dict. Fixed indexing of embeddings with format of {index int: claim str}
* ```batch_size``` : int, optional (default 128).Number of tokens of each training batch
* ```embedding_size``` : int, optional (default 256). Dimentionality of embedding vectors
* ```learning_rate``` : float, optional (default 0.1). Initial leaerning rate of gradient descent optimizer
* ```decay_steps``` : int, optional (default 50000). Step frequency to decay the learning rate. e.g. if 5000, model will reduce learning rate by decay_rate every 5000 trained batches
* ```decay_rate``` : float, optional (default 0.95). Percentage of learning rate decay rate
* ```num_sampled``` : int, optional (default 64). Size of negative samling
* ```valid_size``` : float, optional (default 16). Random set of words to evaluate similarity on

##### Methods

###### _train_

```python
train(data, num_epoch, log_dir, evaluate_every=2000)
```

Parameters
* ```data``` : list. Training index data with format of ```[[1,573,203], [16389,8792], ... [0,4,8394,20094]]```; the index should be consistent with dictionary
* ```num_epoch``` : int. Number of epoch to go over whole dataset
* ```log_dir``` : str. Directory of model logs
* ```evaluate_every``` : int, optional (default: 2000). How many steps the model evluates loss (evaluate sampled validation set on every evaluate_every * 10 steps)

##### Examples
```python
>>> from enid.claim2vec import Claim2Vec
>>> c2v = Claim2Vec(d)
>>> dataset = [[1,573,203], [16389,8792], ... [0,4,8394,20094]]
>>> c2v.train(dataset, 5, 'c2v_logs', 5000)
```

### 3. than_clf

This module defines the framework of Time-Aware Hierarchical Attention Model, along with training and deployment process. It uses an embedding layer, followed by a token-level bi-LSTM with attention, then a sentence-level time-aware-LSTM with attention and then sofrmax layer:

<p align="center">
  <img src="imgs/than.png" class="center" height="700" />
</p>

#### class T_HAN

```
enid.than_clf.T_HAN(mode, **kwargs)
```

##### Parameters

* ```mode``` : str. ```'train'``` or ```'deploy'``` mode

##### Train Mode Parameters (if you choose ```'train'``` mode, please provide:)

* ```max_sequence_length``` : int. The fixed padding number of encounters.
* ```max_sentence_length``` : int. The fixed padding number of claims each encounter
* ```num_classes``` : int. The number of target classes
* ```hidden_size``` : int. The number of LSTM units
* ```pretrain_embedding``` : 2-D numpy array (```vocab_size```, ```embedding_size```). Random initialized embedding matrix
* ```learning_rate``` : float. Initial learning rate for Adam Optimizer
* ```decay_steps``` : int. Step frequency to decay the learning rate. e.g. if 5000, model will reduce learning rate by decay_rate every 5000 trained batches
* ```decay_rate``` : float. Percentage of learning rate decay rate
* ```dropout_keep_prob``` : float. Percentage of neurons to keep from dropout regularization each layer
* ```l2_reg_lambda``` : float, optional (default: .0). L2 regularization lambda for fully-connected layer to prevent potential overfitting
* ```objective``` : str, optional (default: ```'ce'```). The objective function (loss) model trains on; if 'ce', use cross-entropy, if 'auc', use AUROC as objective
* ```initializer``` : tf tensor initializer object, optional (default: ```tf.orthogonal_initializer()```). Initializer for fully connected layer weights

##### Deploy Mode Parameters (if you choose ```'deploy'``` mode, please provide:)

* ```model_path``` : str. The path to store the model
* ```step``` : int, optional (defult ```None```). If not None, load specific model with given step

##### Methods

###### _train_

```python
train(t_train, x_train, y_train, dev_sample_percentage, num_epochs, batch_size, evaluate_every, model_path, debug=False)
```

Parameters

* ```t_train``` : 2-D numpy array, shape (```num_samples```, ```max_sequence_length```). Time deltas of all samples and encounters
* ```x_train``` : 3-D numpy array, shape (```num_samples```,  ```max_sequence_length```, ```max_sentence_length```). Index of all of all samples, encounters and claims
* ```y_train``` : 2-D numpy array, shape (```num_samples```, ```num_classes```). Training ground truth
* ```dev_sample_percentage``` : float. Percentage of ```x_train``` seperated from training process and used for validation
* ```num_epochs``` : int. Mumber of epochs of training, one epoch means finishing training entire training set
* ```batch_size``` : int. Size of training batches, this won't affect training speed significantly; smaller batch leads to more regularization
* ```evaluate_every``` : int. Mumber of steps to perform a evaluation on development (validation) set and print out info
* ```model_path``` : str. The path to store the model

###### _deploy_

```python
deploy(t_test, x_test)
```

Parameters

* ```t_test``` : 2-D numpy array, shape (```num_samples```, ```max_sequence_length```). Time deltas of all samples and encounters
* ```x_test``` : 3-D numpy array, shape (```num_samples```,  ```max_sequence_length```, ```max_sentence_length```). Index of all of all samples, encounters and claims

##### Examples
```python
>>> from enid.than_clf import T_HAN
>>> model_1 = T_HAN('train', max_sequence_length=50, max_sentence_length=20,
        hidden_size=128, num_classes=2, pretrain_embedding=emb,
        learning_rate=0.05, decay_steps=5000, decay_rate=0.9,
        dropout_keep_prob=0.8, l2_reg_lambda=0.0, objective='ce')
>>> model_1.train(t_train=T, x_train=X,
                y_train=y, dev_sample_percentage=0.01,
                num_epochs=20, batch_size=64,
                evaluate_every=100, model_path='./model/')

>>> model_2 = T_HAN('deploy', model_path='./model')
>>> model_2.deploy(t_test=T_test, x_test=X_test)
array([9.9515426e-01,
       4.6948572e-03,
       3.1738445e-02,,
       ...,
       9.9895418e-01,
       5.6348788e-04,
       9.9940193e-01], dtype=float32)
```

### 4. visualization

This module contains tools for plot classification model performance.

#### function plot_performance

```python
enid.visualizations.plot_performance(out_true, out_pred, save_name=None)
```

##### Parameters
* ```out_true``` : list or 1-D array. List of output booleans indicating if True
* ```out_pred``` : list or 1-D array. List of probabilities
* ```save_name``` : str, optional (default: None). The path of output image; if provided, save the plot to disk

##### Returns
1. **ROC (Receiver Operating Characteristic) Curve**: A curve used to define a model’s ability to separate patients who encounter unplanned readmission or adverse events from patients who do not. A model with perfect prediction would have and area of 1.0 under the curve. This is usually the best metric for a machine/deep learning classifier’s relative performance.
2. **Precision-Recall Curve**: This metric depicts the precision vs. recall trade-off. Precision can be translated as ‘percent of the time the model is correct.’ Recall is the total percent of true positives detected, starting from the most probable down to least probable as evaluated by the model. A perfect prediction algorithm has an area of 1.0. This is great for describing the ranking ability of a model.
3. **Precision-Threshold Curve**: This curve helps you choose a threshold when deciding where to operate on your precision recall curve if you are using a threshold.
4. **Calibration Curve**: We want probability values that come out of a deep learning algorithm to properly match up with actual probabilities (e.g. a 0.5 confidence should be right 50% of the time, not 80% of the time). This curve demonstrates that.

##### Examples
```python
>>> from enid.visualizations import plot_performance
>>> y_true = [0, 1, ... 1, 0]
>>> y_prob = [0.0000342, 0.99999974, ... 0.84367323, 0.5400342]
>>> plot_performance(y_true, y_prob, None)
```
<p align="center">
  <img src="imgs/performance.png" class="center" height="200"/>
</p>

#### function plot_comparison

```python
enid.visualizations.plot_comparison(y_true, y_score_1, y_score_2, name_1, name_2, thre=0.5, save_name=None)
```

##### Parameters
* ```y_true``` : list or 1-D array. List of output booleans indicating if True
* ```y_score_1``` : list or 1-D array. List of probabilities of model 1
* ```y_score_2``` : list or 1-D array. List of probabilities of model 2
* ```name_1``` : str. Name of model 1
* ```name_2``` : str. Name of model 2
* ```thre``` : float, optional (default: 0.5). Threshold for point marker on the curves
* ```save_name``` : str, optional (default: None). The path of output image; if provided, save the plot to disk

##### Examples
```python
>>> from enid.visualizations import plot_comparison
>>> y_true = [0, 1, ... 1, 0]
>>> y_prob_1 = [0.0000342, 0.99999974, ... 0.84367323, 0.5400342]
>>> y_prob_2 = [0.0000093, 0.99999742, ... 0.99999618, 0.2400342]
>>> plot_comparison(y_true, y_prob_1, y_prob_2, 'THAN', 'XGBoost')
```
<p align="center">
  <img src="imgs/compare.png" class="center" height="200"/>
</p>

### 5. interpreter

This module contains tools to visualize timelines of samples by interpreting models.

#### function monitor_interpreter

```python
enid.interpreter.monitor_interpreter(data, model_path, step=None, most_recent=20, save_name=None)
```

##### Parameters
* ```data``` : dict. One data sample with format {time: [codes]}, consistent with input for HierarchicalVectorizer.
* ```model_path``` : str. The path to store the model
* ```step``` : int, optional (defult None). If not None, load specific model with given step
* ```most_recent``` : int, optional (default 20). If provided, only plot most_recent timestamps
* ```save_name``` : str, optional (default None). The HTML file path to output interactive visualization; if None, just display

##### Notes
This tools is a patient medical history supervision system with visualizations to highlight those encounters and medical visits that significantly change the risk scores. In other words, the model is re-applied on patient at each new encounter and we are able to monitor "spikes" caused by certain diagnosis, procedures or medicines.

##### Examples
```python
>>> from enid.interpreter import monitor_interpreter
>>> data = {"2015-01-05": ["CPT-97112", "ICD9DX-7813", ...  "CPT-97530",  "POS-11"],
            ...
            "2016-12-18": ["CPT-3078F", "ICD10DX-I10", ... "CPT-99213" ]}
>>> monitor_interpreter(data, model_path="./model", most_recent=30)
```

<p align="center">
  <img src="imgs/monitor.gif" class="center" />
</p>


## For more information

*   [Inovalon Website](https://www.inovalon.com/)
*   [Inovalon AI Solutions](https://www.inovalon.com/solutions/payers/artificial-intelligence/)

## License

[Apache License 2.0](LICENSE)
