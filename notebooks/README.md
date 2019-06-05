# Training FLOPs for one training instance

1. Phase: `training`
2. Batch size: `1`


### Anomaly Detection Models
Model                                                                                  |  #Parameters |   Model Size(MB)  |   gFLOPs (multiply-add)
---------------------------------------------------------------------------------------|--------------|-------------------|--------------------------
[DeepConvLSTMModel](./deep_conv_lstm.ipynb)                                            |  3964754     |   15.859016       |   0.34569984
[LSTM Anomaly Detect](./deepts_models.ipynb#LSTM-Anomaly-Detect)                       |  488501      |   1.954004        |   0.1460163
[Keras Anomaly Detection 02](./deepts_models.ipynb#Keras-Anomaly-Detection)            |  83072       |   0.332288        |   0.041659
[Keras Anomaly Detection 03](./deepts_models.ipynb#MTSAnomalyDetection)                |  166016      |   0.664064        |   0.083319	
[LSTM-FCN](./lstm_fcn.ipynb)                                                           |  1369140     |   5.47656         |   1.582317
[NetTraffic Anomaly Detection (LSTMModel)](./net_traffic_anom_detect_google.ipynb)     |  375551      |   1.502204        |   0.055902
[SMD Anomaly Detection](./smd_anom_detect.ipynb)                                       |  4812288     |   19.249152       |   20.029428

### Fully Connected Models 
Model                                                                                  |  #Parameters |   Model Size(MB)  |   gFLOPs (multiply-add)
---------------------------------------------------------------------------------------|--------------|-------------------|--------------------------
[EnglishAcousticModel](./dlbs_models.ipynb#Training)                                   |  34678784    |   138.715136      |   0.103981
[DeepMNIST](./dlbs_models.ipynb#Training)                                              |  11972510    |   47.89004        |   0.035895

### Convolutional Neural Networks
Model                                                                                  |  #Parameters |   Model Size(MB)  |   gFLOPs (multiply-add)
---------------------------------------------------------------------------------------|--------------|-------------------|--------------------------
[AlexNet](./dlbs_models.ipynb#Training)                                                |  62378344    |   249.513376      |   3.405768
[AlexNetOWT](./dlbs_models.ipynb#Training)                                             |  61100840    |   244.40336       |   2.142565
[VGG11](./dlbs_models.ipynb#Training)                                                  |  132863336   |   531.453344      |   22.82727
[VGG13](./dlbs_models.ipynb#Training)                                                  |  133047848   |   532.191392      |   33.925399
[VGG16](./dlbs_models.ipynb#Training)                                                  |  138357544   |   553.430176      |   46.410793
[VGG19](./dlbs_models.ipynb#Training)                                                  |  143667240   |   574.66896       |   58.896187
[Overfeat](./dlbs_models.ipynb#Training)                                               |  145920872   |   583.683488      |   8.404212
[ResNet18](./dlbs_models.ipynb#Training)                                               |  11703464    |   46.813856       |   5.480755
[ResNet34](./dlbs_models.ipynb#Training)                                               |  21819048    |   87.276192       |   11.029819
[ResNet50](./dlbs_models.ipynb#Training)                                               |  25610152    |   102.440608      |   12.267553
[ResNet101](./dlbs_models.ipynb#Training)                                              |  44654504    |   178.618016      |   23.404216
[ResNet152](./dlbs_models.ipynb#Training)                                              |  60344232    |   241.376928      |   34.54088
[ResNet200](./dlbs_models.ipynb#Training)                                              |  64849832    |   259.399328      |   45.022446
[ResNet269](./dlbs_models.ipynb#Training)                                              |  102326184   |   409.304736      |   60.089696

