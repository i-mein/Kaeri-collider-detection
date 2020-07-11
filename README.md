# Collider-detection-AI using Vibration data
DACON Competition | Korea Atomic Energy Research Institute 

### Description:

[https://dacon.io/competitions/official/235614/overview/](https://dacon.io/competitions/official/235614/overview/)

# Winning Solution documentation

---

#### User/Team name: IME

#### LB score: 
    - Public: 4th 
    - Pvt: 7th 

#### Submission date: 17/07/2020

---

## 1: Library and Data

Use of TF Keras, Pytorch, ....

Read the dataset

```
train_x = pd.read_csv(os.path.join(root_dir, 'train_features.csv'))
train_y = pd.read_csv(os.path.join(root_dir, 'train_target.csv'))
test_x = pd.read_csv(os.path.join(root_dir, 'test_features.csv'))
```

## 2: Data Cleansing & Pre-Processing

## 3: Exploratory Data Analysis

## 4: Feature Engineering & Initial Modeling

- FE lags on various windows (per sensor) —> 24 additional features + 5 raw = `29 features with shape: (-1, 375, 29, 1)`
- FE pct change+ agg stats for each sensor Si  —> `68 features with shape: (-1, 68)`
    - mean, median, min, max, std, percentiles, skew
    - min/max, norm, mean absolute change, absolute max/min, max to min, absolute average

![DACON%20Kaeri%20Readme%20d864e9db0f994d518c14093a77d9474c/Untitled.png](DACON%20Kaeri%20Readme%20d864e9db0f994d518c14093a77d9474c/Untitled.png)

Tried also but didn't work: 

- interactions between S1, S2, S3, S4
- FFT features
- exponential weighted functions

insert figure with features per model

## 5: Model Tuning & Evaluation

### Model 1 [Keras]

CNN2d, 1 channel with 6 layers + FC head with 3 Dense layers

Parameters:

```
filters = [32, 32*2, 32*4, 32*8, 32*16, 32*32]
kernel = (5,1) 
activation = 'elu' 
padding = 'same'
dropout = 0.2
batch_normalization conv layers = True
batch_normalization dense layers = False
dense units = [512, 128, 16]
```

Summary:

```
Model 1 [Keras]: 
_________________________________________________________________
Layer (type)            Output Shape                    Param #
=================================================================
input_26 (InputLayer) [(None, 375, 5, 1)]                    0
_________________________________________________________________
conv2d_150 (Conv2D)   (None, 375, 5, 32)                   192
_________________________________________________________________
batch_normalization_210 (Bat (None, 375, 5, 32)            128
_________________________________________________________________
max_pooling2d_120 (MaxPoolin (None, 187, 5, 32)              0
_________________________________________________________________
conv2d_151 (Conv2D) (None, 187, 5, 64)                   10304
_________________________________________________________________
batch_normalization_211 (Bat (None, 187, 5, 64) 256
_________________________________________________________________
max_pooling2d_121 (MaxPoolin (None, 93, 5, 64) 0
_________________________________________________________________
conv2d_152 (Conv2D) (None, 93, 5, 128) 41088
_________________________________________________________________
batch_normalization_212 (Bat (None, 93, 5, 128) 512
_________________________________________________________________
max_pooling2d_122 (MaxPoolin (None, 46, 5, 128) 0
_________________________________________________________________
conv2d_153 (Conv2D) (None, 46, 5, 256) 164096
_________________________________________________________________
batch_normalization_213 (Bat (None, 46, 5, 256) 1024
_________________________________________________________________
max_pooling2d_123 (MaxPoolin (None, 23, 5, 256) 0
_________________________________________________________________
conv2d_154 (Conv2D) (None, 23, 5, 512) 655872
_________________________________________________________________
batch_normalization_214 (Bat (None, 23, 5, 512) 2048
_________________________________________________________________
max_pooling2d_124 (MaxPoolin (None, 11, 5, 512) 0
_________________________________________________________________
conv2d_155 (Conv2D) (None, 11, 5, 1024) 2622464
_________________________________________________________________
batch_normalization_215 (Bat (None, 11, 5, 1024) 4096
_________________________________________________________________
max_pooling2d_125 (MaxPoolin (None, 5, 5, 1024) 0
_________________________________________________________________
flatten_24 (Flatten) (None, 25600) 0
_________________________________________________________________
dense_99 (Dense) (None, 512) 13107712
_________________________________________________________________
elu_194 (ELU) (None, 512) 0
_________________________________________________________________
dropout_74 (Dropout) (None, 512) 0
_________________________________________________________________
dense_100 (Dense) (None, 128) 65664
_________________________________________________________________
elu_195 (ELU) (None, 128) 0
_________________________________________________________________
dropout_75 (Dropout) (None, 128) 0
_________________________________________________________________
dense_101 (Dense) (None, 16) 2064
_________________________________________________________________
elu_196 (ELU) (None, 16) 0
_________________________________________________________________
dropout_76 (Dropout) (None, 16) 0
_________________________________________________________________
dense_102 (Dense) (None, 4) 68
=================================================================
Total params: 16,677,588
Trainable params: 16,673,556
Non-trainable params: 4,032
_________________________________________________________________
```

### Training Scheme (Model 1)

Optimizer: `Adam(LR=0.1) + SWA`  

LR schedule: `Cyclic LR (exp)`

Early stopping callback with patience 50

Batch size = 256

Train one model for Position (XY), one model for Mass (M) & one model for Velocity (V)

Model 1-XY `validation loss = 0.0018968`

Model 1-M `validation_loss=0.0005028`

Model 1-V `validation loss=3.3e-05`

---

### Model 2 [Pytorch]

CNN2d with 2 channels, 6 layers + FC head with 3 Dense layers

Parameters:

```
filters = [16, 32, 64, 128, 256, 512]
kernel = (5,1) 
activation = 'elu' 
padding = 'same'
dropout = 0.2
batch_normalization conv layers = True
batch_normalization dense layers = False
dense units = [512, 256, 128]
```

### Training Scheme (Model 2)

Channel 1: `raw data + white Noise (mean=0, std=0.001)`

Channel 2: `normalized data + white Noise (mean=0, std=0.001)`

Optimizer: `Adam(LR=0.001)`  

LR schedule: `ReduceLROnPlateau`

Batch size: `256`

Train one model for Position (XY), one model for Mass (M) & one model for Velocity (V) using KFold (5 folds)

Model 1-XY `validation loss = 0.00013199864`

Model 1-M `validation_loss = 0.005548691534` 

Model 1-V `validation loss = 0.00179906`

---

### Model 3 [Keras]

CNN2d + MLP concat + FC head

Parameters:

```
filters = [32, 32*2, 32*4, 32*8, 32*16, 32*32]
kernel = (5,1) 
activation = 'elu' 
padding = 'same'
dropout = 0.2
batch_normalization conv layers = True
batch_normalization dense layers = True
dense units = [512, 256]
fc dense units (after concat) = [1024, 512, 128]
```

Summary: 

```
Model 3 [Keras]:
__________________________________________________________________________________________________
Layer (type) Output Shape Param # Connected to
==================================================================================================
input_6 (InputLayer) [(None, 375, 29, 1)] 0
__________________________________________________________________________________________________
conv2d_18 (Conv2D) (None, 371, 29, 32) 192 input_6[0][0]
__________________________________________________________________________________________________
batch_normalization_21 (BatchNo (None, 371, 29, 32) 128 conv2d_18[0][0]
__________________________________________________________________________________________________
elu_21 (ELU) (None, 371, 29, 32) 0 batch_normalization_21[0][0]
__________________________________________________________________________________________________
max_pooling2d_18 (MaxPooling2D) (None, 185, 29, 32) 0 elu_21[0][0]
__________________________________________________________________________________________________
conv2d_19 (Conv2D) (None, 181, 29, 64) 10304 max_pooling2d_18[0][0]
__________________________________________________________________________________________________
batch_normalization_22 (BatchNo (None, 181, 29, 64) 256 conv2d_19[0][0]
__________________________________________________________________________________________________
elu_22 (ELU) (None, 181, 29, 64) 0 batch_normalization_22[0][0]
__________________________________________________________________________________________________
max_pooling2d_19 (MaxPooling2D) (None, 90, 29, 64) 0 elu_22[0][0]
__________________________________________________________________________________________________
conv2d_20 (Conv2D) (None, 86, 29, 128) 41088 max_pooling2d_19[0][0]
__________________________________________________________________________________________________
batch_normalization_23 (BatchNo (None, 86, 29, 128) 512 conv2d_20[0][0]
__________________________________________________________________________________________________
elu_23 (ELU) (None, 86, 29, 128) 0 batch_normalization_23[0][0]
__________________________________________________________________________________________________
max_pooling2d_20 (MaxPooling2D) (None, 43, 29, 128) 0 elu_23[0][0]
__________________________________________________________________________________________________
conv2d_21 (Conv2D) (None, 39, 29, 256) 164096 max_pooling2d_20[0][0]
__________________________________________________________________________________________________
batch_normalization_24 (BatchNo (None, 39, 29, 256) 1024 conv2d_21[0][0]
__________________________________________________________________________________________________
elu_24 (ELU) (None, 39, 29, 256) 0 batch_normalization_24[0][0]
__________________________________________________________________________________________________
max_pooling2d_21 (MaxPooling2D) (None, 19, 29, 256) 0 elu_24[0][0]
__________________________________________________________________________________________________
conv2d_22 (Conv2D) (None, 15, 29, 512) 655872 max_pooling2d_21[0][0]
__________________________________________________________________________________________________
batch_normalization_25 (BatchNo (None, 15, 29, 512) 2048 conv2d_22[0][0]
__________________________________________________________________________________________________
elu_25 (ELU) (None, 15, 29, 512) 0 batch_normalization_25[0][0]
__________________________________________________________________________________________________
max_pooling2d_22 (MaxPooling2D) (None, 7, 29, 512) 0 elu_25[0][0]
__________________________________________________________________________________________________
conv2d_23 (Conv2D) (None, 3, 29, 1024) 2622464 max_pooling2d_22[0][0]
__________________________________________________________________________________________________
input_7 (InputLayer) [(None, 68)] 0
__________________________________________________________________________________________________
batch_normalization_26 (BatchNo (None, 3, 29, 1024) 4096 conv2d_23[0][0]
__________________________________________________________________________________________________
dense_8 (Dense) (None, 512) 35328 input_7[0][0]
__________________________________________________________________________________________________
elu_26 (ELU) (None, 3, 29, 1024) 0 batch_normalization_26[0][0]
__________________________________________________________________________________________________
dropout_7 (Dropout) (None, 512) 0 dense_8[0][0]
__________________________________________________________________________________________________
max_pooling2d_23 (MaxPooling2D) (None, 1, 29, 1024) 0 elu_26[0][0]
__________________________________________________________________________________________________
dense_9 (Dense) (None, 256) 131328 dropout_7[0][0]
__________________________________________________________________________________________________
flatten_2 (Flatten) (None, 29696) 0 max_pooling2d_23[0][0]
__________________________________________________________________________________________________
dropout_8 (Dropout) (None, 256) 0 dense_9[0][0]
__________________________________________________________________________________________________
concatenate_2 (Concatenate) (None, 29952) 0 flatten_2[0][0] dropout_8[0][0]
__________________________________________________________________________________________________
flatten_3 (Flatten) (None, 29952) 0 concatenate_2[0][0]
__________________________________________________________________________________________________
dense_10 (Dense) (None, 1024) 30671872 flatten_3[0][0]
__________________________________________________________________________________________________
batch_normalization_27 (BatchNo (None, 1024) 4096 dense_10[0][0]
__________________________________________________________________________________________________
elu_27 (ELU) (None, 1024) 0 batch_normalization_27[0][0]
__________________________________________________________________________________________________
dropout_9 (Dropout) (None, 1024) 0 elu_27[0][0]
__________________________________________________________________________________________________
dense_11 (Dense) (None, 512) 524800 dropout_9[0][0]
__________________________________________________________________________________________________
batch_normalization_28 (BatchNo (None, 512) 2048 dense_11[0][0]
__________________________________________________________________________________________________
elu_28 (ELU) (None, 512) 0 batch_normalization_28[0][0]
__________________________________________________________________________________________________
dropout_10 (Dropout) (None, 512) 0 elu_28[0][0]
__________________________________________________________________________________________________
dense_12 (Dense) (None, 128) 65664 dropout_10[0][0]
__________________________________________________________________________________________________
batch_normalization_29 (BatchNo (None, 128) 512 dense_12[0][0]
__________________________________________________________________________________________________
elu_29 (ELU) (None, 128) 0 batch_normalization_29[0][0]
__________________________________________________________________________________________________
dropout_11 (Dropout) (None, 128) 0 elu_29[0][0]
__________________________________________________________________________________________________
dense_13 (Dense) (None, 4) 516 dropout_11[0][0]
==================================================================================================
Total params: 34,938,244
Trainable params: 34,930,884
Non-trainable params: 7,360
__________________________________________________________________________________________________
```

### Training scheme (Model 3)

Optimizer: `Adam(LR=0.01)`  

LR schedule: `step decay`

Early stopping callback with patience 50

Batch size = 256

Train one model for Position (XY), one model for Mass (M) & one model for Velocity (V)

Model 1-XY `validation loss = 0.000263`

Model 1-M `validation_loss = 0.000038` 

Model 1-V `validation loss = 0.00012` 

---

### Ensemble

```
X Prediction: (Model 1 [XY] + Model 2 [XY] + Model 3 [XY])/3 
Y Prediction: (Model 1 [XY] + Model 2 [XY] + Model 3 [XY])/3 
M Prediction: (Model 1 [M] + Model 2 [M] + Model 3 [M])/3 
V Prediction: (Model 1 [V] + Model 2 [V] + Model 3 [V])/3 
```

### Metric/Loss

KAERI metric

```python
def kaeri_metric(y_true,  y_pred):

    '''
        y_true: dataframe with true values of X,Y,M,V
        y_pred: dataframe with pred values of X,Y,M,V

        return: KAERI metric
    '''
    t1, p1 = np.array(y_true)[:,:2], np.array(y_pred)[:,:2]
    E1 = np.mean(np.sum(np.square(t1 - p1), axis=1) / 2e+04)
      
    t2, p2 = np.array(y_true)[:,2:], np.array(y_pred)[:,2:]
    E2 = np.mean(np.sum(np.square((t2 - p2) / (t2 + 1e-06)), axis=1))

    return 0.5*E1 + 0.5*E2
```

## 6. Conclusion & Discussion

To my best of knowledge a single model with proper features and tuned parameters can outperform a very complex models with complex features.

Next steps (that I didn't have time to try during the submission time)

- extract FFT features from various bands & from correlations between them
- Mel spectrograms, MFCC features from spectra
- Kalman filtering
- FE + Feature selection

    To select best features amongst the pool of features

- Hyperparameter tuning (concentrate on a single model and tune its parameters using W&B)

 

# Discussions

# Kernels

[SHIFTED-RFC Pipeline](https://www.kaggle.com/sggpls/shifted-rfc-pipeline)

# Resources

[basveeling/wavenet](https://github.com/basveeling/wavenet)

[tf.data: Build TensorFlow input pipelines | TensorFlow Core](https://www.tensorflow.org/guide/data)

[](https://arxiv.org/pdf/1807.03247.pdf)

# Exploring The Dataset

The First Important thing i found that there was null values in the dataset. By running `dataset.isnull().sum()`

- There is not that much NaN values

# Data Visualisations

Data Visualisations is always a fun part, it gives us a LOT of insights about the dataset. Soo why not to do this

# Evaluation Metric

SMAPE  


# Next steps (that I didn't have time to try during the submission time)

- extract FFT features from various bands & from correlations between them
- Mel spectrograms, MFCC features from spectra
- Kalman filtering
- FE + Feature Selection
- Hyperparameter tuning (Weights & Biases)



# Weights & Biases Intro (in case you wish to experiment)

1. Install It 

```python
!pip install --upgrade wandb
```

2. Login from your ID

```python
!wandb login {secret_value_0}
```

3. Intialise your project (and hyperparameters)

```python
import wandb
from wandb.keras import WandbCallback

defaults=dict(
    dropout = 0.2,
    learn_rate = 0.01,
    beta1 = 0.9,
    beta2 = 0.999,
    epsilon = 1e-07,
    label_smooth = 0.1
    )

# Initialize a new wandb run and pass in the config object
# wandb.init(anonymous='allow', project="kaggle", config=defaults)
wandb.init(project="visualize-models", config=defaults, name="neural_network")
config = wandb.config 
```

4. Add parameters in your model and in compile like this——

```python
with strategy.scope():
    model = tf.keras.Sequential([
        efn.EfficientNetB5(
            input_shape=(*IMAGE_SIZE, 3),
            weights='imagenet',
            include_top=False
        ),
        L.GlobalAveragePooling2D(),
        L.Dense(1024, activation = 'relu'), 
        L.Dropout(config.dropout), 
        L.Dense(512, activation= 'relu'), 
        L.Dropout(config.dropout), 
        L.Dense(256, activation='relu'), 
        L.Dropout(config.dropout), 
        L.Dense(128, activation='relu'), 
        L.Dropout(config.dropout), 
        L.Dense(1, activation='sigmoid')
    ])
```

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(
    learning_rate=config.learn_rate,
    beta_1=config.beta1,
    beta_2=config.beta2,
    epsilon=config.epsilon),
    #loss = 'binary_crossentropy',
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = config.label_smooth),
    metrics=['binary_crossentropy', 'accuracy']
)
model.summary()
```

5. Add the callback 

```python
labels=["benign","malignant"]

history = model.fit(
    get_training_dataset(), 
    epochs=EPOCHS, 
    validation_data=get_validation_dataset(),
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=[WandbCallback(data_type="image", labels=labels)]
)
```

Maybe this also 

[https://www.wandb.com/articles/hyperparameter-tuning-as-easy-as-1-2-3](https://www.wandb.com/articles/hyperparameter-tuning-as-easy-as-1-2-3)

 





