# Collider-detection-AI using Vibration data
DACON Competition | Korea Atomic Energy Research Institute 

### Description:

Collider Detection AI for early diagnosis of faults in Nuclear Power Plants using Vibration data (LB #7th)

[https://dacon.io/competitions/official/235614/overview/](https://dacon.io/competitions/official/235614/overview/)

---

### Files:

If you wish to experiment with the models, download data from competition site and use either the main file (ensemble of 3 models) or experiment with specific models. The training schemes and architectures are described below.   

- `KAERI_source_code_IME.ipynb`: single file with whole code to train all 3 models and create the final submission file.  
- `KAERI CNN2d Keras xxx.ipynb`: TF_Keras training CNN2d model 
- `KAERI CNN2d Torch xxx.ipynb`: Pytorch training CNN2d model 
- `KAERI CNN2d-MLP Keras xxx.ipynb`: TF_Keras training CNN2d + MLP model concat (with sequence and tabular data)
- `KAERI ensemble.ipynb`: ensemble of 3 best models 


---

# Winning Solution documentation

---

#### User/Team name: IME

#### LB Position: 
    - Public: 4th  | LB: 0.0037xx 
    - Pvt: 7th     | LB: 0.0042xx
---

## 1: Library and Data

- 4 acceleration sensors 
- ~0.0015 sec measurements with 25600 Hz sampling --> 375 points per measurement
- 2800 collider ids in training set
- 4 target variables: position (X,Y), mass (M) and velocity (V) of the collider

- Use of TF Keras, Pytorch, scikit-learn, scipy and more

- Evaluation Metric: SMAPE  


## 2: Data Cleansing & Pre-Processing

## 3: Exploratory Data Analysis

to be added soon

## 4: Feature Engineering & Initial Modeling

- FE lags on various windows (per sensor) —> 24 additional features + 5 raw = `29 features with shape: (-1, 375, 29, 1)`
- FE pct change+ agg stats for each sensor Si  —> `68 features with shape: (-1, 68)`
    - mean, median, min, max, std, percentiles, skew
    - min/max, norm, mean absolute change, absolute max/min, max to min, absolute average

![](Untitled.png)


Tried also but didn't work: 

- interactions between S1, S2, S3, S4
- FFT features
- exponential weighted functions


## 5: Model Tuning & Evaluation


### Model 1 [Pytorch]

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

### Training Scheme (Model 1)

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




### Model 2 [Keras]

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


### Training Scheme (Model 2)

Optimizer: `Adam(LR=0.1) + SWA`  

LR schedule: `Cyclic LR (exp)`

Early stopping callback with patience 50

Batch size = 256

Train one model for Position (XY), one model for Mass (M) & one model for Velocity (V)

Model 1-XY `validation loss = 0.0018968`

Model 1-M `validation_loss=0.0005028`

Model 1-V `validation loss=3.3e-05`

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

My intention here was to experiment with different frameworks (TF, Pytorch) and make comparisons on the model performance, pipelines etc

I was experimenting with many architectures but at the end I didn't have time to focus on a single one and tune it properly and decide it to 
spend the final days to ensembling. Hence, I picked the best 3 models from the 'experimental pool' to boost my LB scores.

However, to my best of knowledge a single model with proper features and tuned parameters can outperform very complex models with complex features. 
Next steps will be towards that direction. 


### Next steps (that I didn't have time to try during the submission time)

- extract FFT features from various bands & from correlations between them
- Mel spectrograms, MFCC features from spectra
- Kalman filtering
- FE + Feature Selection (To select best features amongst the pool of features)
- Hyperparameter tuning (Weights & Biases)




