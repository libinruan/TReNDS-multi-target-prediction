#### LOG

6/18 First NN

6/19 look at external dataset

Giving less importance to FNC features since they are easier to overfit due to high dimensionality.

#### Intro

[TReNDS Neuroimaging](https://www.kaggle.com/anshumoudgil/brain-s-network-activation-via-eda)

以多變量`空間信號源為基礎的形態學`（source-based morphometry, SBM）是利用獨立成分分析方法（independent component analysis, ICA）解碼大腦灰質結構的空間分佈和負荷係數。負荷係數（loading coefficients）代表在這個兩個群體個體大腦灰質結構空間分佈的`體積大小`或`濃度比例`。

#### The dataset

##### External Dataset

https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection

https://github.com/Tencent/MedicalNet

https://github.com/kenshohara/3D-ResNets-PyTorch

https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection/tree/master/3DNet



##### Original Dataset

- **fMRI_train** - a folder containing 53 3D spatial maps for train samples in [.mat] format.
- **fMRI_test** - a folder containing 53 3D spatial maps for test samples in [.mat] format.
- **fnc.csv** - static FNC correlation features for both train and test samples.
- **loading.csv** - sMRI SBM loadings for both train and test samples.
- **train_scores.csv** - age and assessment values for train samples.
- **reveal_ID_site2.csv** - a list of subject IDs whose data was collected with a different scanner than the train samples.
- **fMRI_mask.nii** - a 3D binary spatial map.
- **ICN_numbers.txt** - intrinsic connectivity network numbers for each fMRI spatial map; matches FNC names.



#### What is Neuroimaging?

- Structural imaging, which deals with the structure of the nervous system and the diagnosis of gross (large scale) intracranial disease (such as a tumor) and injury.
- Functional imaging, which is used to diagnose metabolic diseases and lesions on a finer scale (such as Alzheimer's disease) and also for neurological and cognitive psychology research and building brain-computer interfaces.

---

#### Model

##### Architecture

https://www.kaggle.com/mercury01/pca-base-keras-nn-nae-metric by Alexandr Osherov (OK)

[Using TabNet - Attention Based Neural Network](https://www.kaggle.com/c/trends-assessment-prediction/discussion/154165) by Nirjhar Roy

[[TF.Keras] Melanoma Classification Starter, TabNet](https://www.kaggle.com/ipythonx/tf-keras-melanoma-classification-starter-tabnet)

[TReNDS Google TabNet Baseline](https://www.kaggle.com/phoenix9032/trends-google-tabnet-baseline)

[3D CNN with keras](https://www.kaggle.com/hrfhgrthdyrd/3d-cnn-with-keras) performance subject to kaggle provision limit

[BaggingRegressor + RAPIDS Ensemble](https://www.kaggle.com/andypenrose/baggingregressor-rapids-ensemble) 0.1593

[RAPIDS Ensemble for TReNDS Neuroimaging](https://www.kaggle.com/tunguz/rapids-ensemble-for-trends-neuroimaging) 0.1595



##### Feature Engineering

[TReNDS: EDA, LightGBM, Rapids.AI SVM](https://www.kaggle.com/rftexas/trends-eda-lightgbm-rapids-ai-svm)



##### Submission

###### No mat file involved

https://www.kaggle.com/kmatsuyama/simple-nn-baseline-using-keras by Kmatsuyama



```
Model = Sequential()
Model.add(Dense(100, input_shape = (445,), activation ='elu')) 
Model.add(Dropout(0.1))

Model.add(Dense(800, activation='elu'))
Model.add(Dropout(0.05))

Model.add(Dense(600, activation='elu'))
Model.add(Dropout(0.1))

Model.add(Dense(400, activation='elu'))
Model.add(Dropout(0.1))

Model.add(Dense(5, activation = 'elu'))

Model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.0001), metrics=[weighted_NAE])
Model.summary()

# 1584
```

```
Model = Sequential()
Model.add(Dense(100, input_shape = (445,), activation ='elu')) 
Model.add(Dropout(0.1))

Model.add(Dense(1000, activation='elu'))
Model.add(Dropout(0.1))

Model.add(Dense(1000, activation='elu'))
Model.add(Dropout(0.1))

Model.add(Dense(400, activation='elu'))
Model.add(Dropout(0.1))

Model.add(Dense(5, activation = 'elu'))

Model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.0001), metrics=[weighted_NAE])
Model.summary()

# 1583
```

```
Model = Sequential()
Model.add(Dense(100, input_shape = (445,), activation ='elu')) 
Model.add(Dropout(0.1))

Model.add(Dense(200, activation='elu'))
Model.add(Dropout(0.1))

Model.add(Dense(300, activation='elu'))
Model.add(Dropout(0.1))

Model.add(Dense(400, activation='elu'))
Model.add(Dropout(0.1))

Model.add(Dense(500, activation='elu'))
Model.add(Dropout(0.1))

Model.add(Dense(600, activation='elu'))
Model.add(Dropout(0.1))

Model.add(Dense(800, activation='elu'))
Model.add(Dropout(0.1))

Model.add(Dense(1000, activation='elu'))
Model.add(Dropout(0.1))

Model.add(Dense(5, activation = 'elu'))

Model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.0001), metrics=[weighted_NAE])
Model.summary()

# 1581, 1574
```



---

#### Site

```
'DMN(17)_vs_SCN(98)',
 'CON(55)_vs_SCN(45)',
 'DMN(32)_vs_SMN(27)',
 'SMN(80)_vs_SMN(54)',
 'CON(79)_vs_SMN(54)',
 'CON(88)_vs_SMN(54)',
 'CON(81)_vs_SMN(54)',
 'CON(79)_vs_SMN(66)',
 'DMN(40)_vs_SMN(80)',
 'DMN(23)_vs_VSN(16)',
 'CON(55)_vs_CON(33)',
 'DMN(17)_vs_CON(33)',
 'CON(88)_vs_CON(43)',
 'DMN(94)_vs_CON(63)',
 'DMN(40)_vs_CON(88)',
 'DMN(17)_vs_CON(88)',
 'CON(83)_vs_CON(37)',
 'DMN(32)_vs_CON(37)',
 'CON(83)_vs_CON(67)',
 'CBN(4)_vs_CON(38)',
 'DMN(17)_vs_DMN(40)',
 'DMN(51)_vs_DMN(40)']
```

