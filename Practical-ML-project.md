## Executive summary

Three different method were used to identify which activity was done
based on sensors data: GBM, rain forest and neural network. GBM and
random forest both gave very results with “just” a 3-fold cross
validation method. Neural network had poor accuracy (approx. 40%).
Eventually, the random forest method was chosen as it gave an accuracy
of more than 99% on a dedicated test set.

## Data loading / Pre-processing

Data is loaded from the .csv files. Then some of the rows are
transformed as they obviously contain factors.

``` r
library(caret)
```

    ## Warning: package 'caret' was built under R version 4.0.4

    ## Loading required package: lattice

    ## Loading required package: ggplot2

``` r
set.seed(070321)

training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")

training$user_name <- as.factor(training$user_name)
training$classe <- as.factor(training$classe)
testing$user_name <- as.factor(testing$user_name)
```

Some cleanup actions are conducted - First 7 columns are removed as they
do not bring any added value to the model - Missing values are replaced
by NA - \#DIV/0! values are replaced by NA - Rows where values contain
more than 10k NA are removed (technically, they contain only NA)

``` r
training <- training[,8:160]

training[training == ""] <- NA
training[training == "#DIV/0!"] <- NA

training <- training[,lapply(training,function(x) sum(is.na(x))) < 10000]
```

Then, subset is create to have my own training / test set (as the test
set provided does not include any results to be compared with). I chose
to have 80% of the data in my training, the rest in the test

``` r
inTrain <- createDataPartition(y = training$classe, p = 0.8, list = FALSE)
myTraining <- training[inTrain,]
myTesting <- training[-inTrain,]
```

## Classification

For the classification, I tried 3 classical appraoachs: Generalized
Boosted Regression, Random Forest and. All of these three can be
supported by the train function of the caret package which make their
implementation easy.

### Transverse settings

Choice was made to use 3-fold cross validation to get good results while
still having good performance. This will be used on all trainings

``` r
ctrl <- trainControl(method="cv", number=3, verbose = FALSE)
```

### GBM

GBM approach is first used:

``` r
fit_gbm <- train(classe ~., method = "gbm", data = myTraining, trControl = ctrl)
```

    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1250
    ##      2        1.5231             nan     0.1000    0.0896
    ##      3        1.4635             nan     0.1000    0.0660
    ##      4        1.4198             nan     0.1000    0.0518
    ##      5        1.3846             nan     0.1000    0.0511
    ##      6        1.3515             nan     0.1000    0.0404
    ##      7        1.3245             nan     0.1000    0.0407
    ##      8        1.2991             nan     0.1000    0.0381
    ##      9        1.2758             nan     0.1000    0.0300
    ##     10        1.2551             nan     0.1000    0.0294
    ##     20        1.1003             nan     0.1000    0.0179
    ##     40        0.9292             nan     0.1000    0.0093
    ##     60        0.8189             nan     0.1000    0.0054
    ##     80        0.7375             nan     0.1000    0.0058
    ##    100        0.6753             nan     0.1000    0.0031
    ##    120        0.6236             nan     0.1000    0.0027
    ##    140        0.5792             nan     0.1000    0.0028
    ##    150        0.5604             nan     0.1000    0.0028
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1850
    ##      2        1.4899             nan     0.1000    0.1306
    ##      3        1.4053             nan     0.1000    0.1039
    ##      4        1.3393             nan     0.1000    0.0841
    ##      5        1.2848             nan     0.1000    0.0795
    ##      6        1.2349             nan     0.1000    0.0624
    ##      7        1.1938             nan     0.1000    0.0561
    ##      8        1.1568             nan     0.1000    0.0482
    ##      9        1.1248             nan     0.1000    0.0550
    ##     10        1.0913             nan     0.1000    0.0384
    ##     20        0.8940             nan     0.1000    0.0188
    ##     40        0.6860             nan     0.1000    0.0128
    ##     60        0.5538             nan     0.1000    0.0065
    ##     80        0.4636             nan     0.1000    0.0047
    ##    100        0.4002             nan     0.1000    0.0027
    ##    120        0.3466             nan     0.1000    0.0027
    ##    140        0.3064             nan     0.1000    0.0029
    ##    150        0.2870             nan     0.1000    0.0018
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2349
    ##      2        1.4589             nan     0.1000    0.1604
    ##      3        1.3555             nan     0.1000    0.1364
    ##      4        1.2692             nan     0.1000    0.0983
    ##      5        1.2066             nan     0.1000    0.0959
    ##      6        1.1460             nan     0.1000    0.0690
    ##      7        1.1003             nan     0.1000    0.0654
    ##      8        1.0587             nan     0.1000    0.0602
    ##      9        1.0194             nan     0.1000    0.0501
    ##     10        0.9869             nan     0.1000    0.0565
    ##     20        0.7560             nan     0.1000    0.0246
    ##     40        0.5306             nan     0.1000    0.0129
    ##     60        0.4034             nan     0.1000    0.0057
    ##     80        0.3223             nan     0.1000    0.0042
    ##    100        0.2643             nan     0.1000    0.0033
    ##    120        0.2225             nan     0.1000    0.0015
    ##    140        0.1878             nan     0.1000    0.0013
    ##    150        0.1742             nan     0.1000    0.0014
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1308
    ##      2        1.5219             nan     0.1000    0.0878
    ##      3        1.4642             nan     0.1000    0.0679
    ##      4        1.4199             nan     0.1000    0.0529
    ##      5        1.3843             nan     0.1000    0.0495
    ##      6        1.3515             nan     0.1000    0.0399
    ##      7        1.3259             nan     0.1000    0.0366
    ##      8        1.3014             nan     0.1000    0.0342
    ##      9        1.2796             nan     0.1000    0.0335
    ##     10        1.2565             nan     0.1000    0.0298
    ##     20        1.1044             nan     0.1000    0.0194
    ##     40        0.9317             nan     0.1000    0.0066
    ##     60        0.8237             nan     0.1000    0.0068
    ##     80        0.7434             nan     0.1000    0.0052
    ##    100        0.6798             nan     0.1000    0.0033
    ##    120        0.6310             nan     0.1000    0.0026
    ##    140        0.5876             nan     0.1000    0.0025
    ##    150        0.5676             nan     0.1000    0.0014
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1847
    ##      2        1.4879             nan     0.1000    0.1291
    ##      3        1.4040             nan     0.1000    0.1002
    ##      4        1.3397             nan     0.1000    0.0899
    ##      5        1.2824             nan     0.1000    0.0704
    ##      6        1.2366             nan     0.1000    0.0612
    ##      7        1.1973             nan     0.1000    0.0649
    ##      8        1.1568             nan     0.1000    0.0550
    ##      9        1.1225             nan     0.1000    0.0497
    ##     10        1.0918             nan     0.1000    0.0392
    ##     20        0.8955             nan     0.1000    0.0300
    ##     40        0.6855             nan     0.1000    0.0161
    ##     60        0.5526             nan     0.1000    0.0053
    ##     80        0.4652             nan     0.1000    0.0043
    ##    100        0.4016             nan     0.1000    0.0035
    ##    120        0.3505             nan     0.1000    0.0019
    ##    140        0.3091             nan     0.1000    0.0021
    ##    150        0.2912             nan     0.1000    0.0017
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2337
    ##      2        1.4603             nan     0.1000    0.1638
    ##      3        1.3572             nan     0.1000    0.1323
    ##      4        1.2732             nan     0.1000    0.1010
    ##      5        1.2096             nan     0.1000    0.0861
    ##      6        1.1557             nan     0.1000    0.0727
    ##      7        1.1100             nan     0.1000    0.0599
    ##      8        1.0706             nan     0.1000    0.0647
    ##      9        1.0303             nan     0.1000    0.0650
    ##     10        0.9900             nan     0.1000    0.0514
    ##     20        0.7573             nan     0.1000    0.0266
    ##     40        0.5275             nan     0.1000    0.0108
    ##     60        0.4017             nan     0.1000    0.0064
    ##     80        0.3226             nan     0.1000    0.0036
    ##    100        0.2645             nan     0.1000    0.0039
    ##    120        0.2223             nan     0.1000    0.0030
    ##    140        0.1877             nan     0.1000    0.0014
    ##    150        0.1743             nan     0.1000    0.0007
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1269
    ##      2        1.5252             nan     0.1000    0.0835
    ##      3        1.4671             nan     0.1000    0.0689
    ##      4        1.4224             nan     0.1000    0.0548
    ##      5        1.3868             nan     0.1000    0.0516
    ##      6        1.3527             nan     0.1000    0.0373
    ##      7        1.3282             nan     0.1000    0.0418
    ##      8        1.3016             nan     0.1000    0.0330
    ##      9        1.2803             nan     0.1000    0.0322
    ##     10        1.2600             nan     0.1000    0.0326
    ##     20        1.1004             nan     0.1000    0.0176
    ##     40        0.9329             nan     0.1000    0.0111
    ##     60        0.8205             nan     0.1000    0.0069
    ##     80        0.7388             nan     0.1000    0.0042
    ##    100        0.6750             nan     0.1000    0.0034
    ##    120        0.6247             nan     0.1000    0.0026
    ##    140        0.5802             nan     0.1000    0.0024
    ##    150        0.5612             nan     0.1000    0.0022
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1834
    ##      2        1.4894             nan     0.1000    0.1263
    ##      3        1.4065             nan     0.1000    0.1039
    ##      4        1.3396             nan     0.1000    0.0885
    ##      5        1.2823             nan     0.1000    0.0700
    ##      6        1.2376             nan     0.1000    0.0677
    ##      7        1.1935             nan     0.1000    0.0547
    ##      8        1.1581             nan     0.1000    0.0558
    ##      9        1.1233             nan     0.1000    0.0464
    ##     10        1.0948             nan     0.1000    0.0456
    ##     20        0.8925             nan     0.1000    0.0243
    ##     40        0.6727             nan     0.1000    0.0099
    ##     60        0.5480             nan     0.1000    0.0050
    ##     80        0.4593             nan     0.1000    0.0051
    ##    100        0.3947             nan     0.1000    0.0032
    ##    120        0.3398             nan     0.1000    0.0028
    ##    140        0.3017             nan     0.1000    0.0024
    ##    150        0.2825             nan     0.1000    0.0023
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2299
    ##      2        1.4620             nan     0.1000    0.1598
    ##      3        1.3600             nan     0.1000    0.1215
    ##      4        1.2827             nan     0.1000    0.1100
    ##      5        1.2132             nan     0.1000    0.0955
    ##      6        1.1536             nan     0.1000    0.0819
    ##      7        1.1035             nan     0.1000    0.0581
    ##      8        1.0642             nan     0.1000    0.0688
    ##      9        1.0210             nan     0.1000    0.0615
    ##     10        0.9822             nan     0.1000    0.0477
    ##     20        0.7528             nan     0.1000    0.0253
    ##     40        0.5251             nan     0.1000    0.0123
    ##     60        0.4002             nan     0.1000    0.0068
    ##     80        0.3195             nan     0.1000    0.0033
    ##    100        0.2611             nan     0.1000    0.0038
    ##    120        0.2190             nan     0.1000    0.0023
    ##    140        0.1862             nan     0.1000    0.0011
    ##    150        0.1722             nan     0.1000    0.0009
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2392
    ##      2        1.4603             nan     0.1000    0.1596
    ##      3        1.3583             nan     0.1000    0.1266
    ##      4        1.2793             nan     0.1000    0.1118
    ##      5        1.2101             nan     0.1000    0.0838
    ##      6        1.1572             nan     0.1000    0.0809
    ##      7        1.1080             nan     0.1000    0.0621
    ##      8        1.0678             nan     0.1000    0.0634
    ##      9        1.0291             nan     0.1000    0.0537
    ##     10        0.9941             nan     0.1000    0.0570
    ##     20        0.7621             nan     0.1000    0.0254
    ##     40        0.5410             nan     0.1000    0.0135
    ##     60        0.4159             nan     0.1000    0.0062
    ##     80        0.3288             nan     0.1000    0.0042
    ##    100        0.2718             nan     0.1000    0.0053
    ##    120        0.2280             nan     0.1000    0.0023
    ##    140        0.1934             nan     0.1000    0.0021
    ##    150        0.1799             nan     0.1000    0.0012

``` r
confusionMatrix(predict(fit_gbm, myTesting), myTesting$class)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1101   22    0    0    0
    ##          B   10  721   16    1    9
    ##          C    1   15  659   30    9
    ##          D    4    1    8  607   12
    ##          E    0    0    1    5  691
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.9633         
    ##                  95% CI : (0.9569, 0.969)
    ##     No Information Rate : 0.2845         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.9536         
    ##                                          
    ##  Mcnemar's Test P-Value : NA             
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9866   0.9499   0.9635   0.9440   0.9584
    ## Specificity            0.9922   0.9886   0.9830   0.9924   0.9981
    ## Pos Pred Value         0.9804   0.9524   0.9230   0.9604   0.9914
    ## Neg Pred Value         0.9946   0.9880   0.9922   0.9891   0.9907
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2807   0.1838   0.1680   0.1547   0.1761
    ## Detection Prevalence   0.2863   0.1930   0.1820   0.1611   0.1777
    ## Balanced Accuracy      0.9894   0.9693   0.9732   0.9682   0.9783

As one can see, results is an accuracy of more than 96% which first
seems to be good.

### Random forest approach

Random forest approach is applied using the exact same parameters

``` r
fit_rf <- train(classe ~., method = "rf", data = myTraining, trControl = ctrl)
confusionMatrix(predict(fit_rf, myTesting), myTesting$class)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1116    4    0    0    0
    ##          B    0  755    2    0    0
    ##          C    0    0  680   19    0
    ##          D    0    0    2  623    0
    ##          E    0    0    0    1  721
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9929          
    ##                  95% CI : (0.9897, 0.9953)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.991           
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9947   0.9942   0.9689   1.0000
    ## Specificity            0.9986   0.9994   0.9941   0.9994   0.9997
    ## Pos Pred Value         0.9964   0.9974   0.9728   0.9968   0.9986
    ## Neg Pred Value         1.0000   0.9987   0.9988   0.9939   1.0000
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2845   0.1925   0.1733   0.1588   0.1838
    ## Detection Prevalence   0.2855   0.1930   0.1782   0.1593   0.1840
    ## Balanced Accuracy      0.9993   0.9970   0.9941   0.9841   0.9998

Accuracy is exceptionaly high, more than 99%.

### Neural network approach

Finally, the neural network approach was used to see its results
(curiosity):

``` r
fit_nnet <- train(classe ~., method = "nnet", data = myTraining, trControl = ctrl)
```

    ## # weights:  63
    ## initial  value 17548.428242 
    ## iter  10 value 16075.064207
    ## iter  20 value 15768.389089
    ## iter  30 value 15501.618376
    ## iter  40 value 15434.034434
    ## iter  50 value 15399.898836
    ## iter  60 value 15353.560970
    ## iter  70 value 15350.948310
    ## iter  80 value 15342.897734
    ## iter  90 value 15287.046008
    ## iter 100 value 15241.566797
    ## final  value 15241.566797 
    ## stopped after 100 iterations
    ## # weights:  179
    ## initial  value 19503.970806 
    ## iter  10 value 16662.031917
    ## iter  20 value 16424.585713
    ## iter  30 value 16307.156576
    ## iter  40 value 15683.420029
    ## iter  50 value 15559.884097
    ## iter  60 value 15496.668721
    ## iter  70 value 15295.063966
    ## iter  80 value 15256.038354
    ## iter  90 value 15192.222473
    ## iter 100 value 15094.482783
    ## final  value 15094.482783 
    ## stopped after 100 iterations
    ## # weights:  295
    ## initial  value 18719.654735 
    ## iter  10 value 15977.689084
    ## iter  20 value 15370.221484
    ## iter  30 value 14995.432562
    ## iter  40 value 14657.145896
    ## iter  50 value 14456.398143
    ## iter  60 value 14132.107179
    ## iter  70 value 13993.486739
    ## iter  80 value 13822.472450
    ## iter  90 value 13707.680282
    ## iter 100 value 13641.458724
    ## final  value 13641.458724 
    ## stopped after 100 iterations
    ## # weights:  63
    ## initial  value 17915.169457 
    ## iter  10 value 16620.669458
    ## iter  20 value 16601.494394
    ## iter  30 value 16592.390188
    ## iter  40 value 16559.037047
    ## iter  50 value 16456.696149
    ## iter  60 value 16440.692498
    ## iter  70 value 16356.672238
    ## iter  80 value 16331.893316
    ## iter  90 value 16288.714327
    ## iter 100 value 15926.905254
    ## final  value 15926.905254 
    ## stopped after 100 iterations
    ## # weights:  179
    ## initial  value 18818.302070 
    ## iter  10 value 16184.109753
    ## iter  20 value 15596.231469
    ## iter  30 value 15313.175523
    ## iter  40 value 15273.618191
    ## iter  50 value 15188.810652
    ## iter  60 value 15137.755019
    ## iter  70 value 15071.511703
    ## iter  80 value 15026.957913
    ## iter  90 value 15020.588551
    ## iter 100 value 15001.858080
    ## final  value 15001.858080 
    ## stopped after 100 iterations
    ## # weights:  295
    ## initial  value 17948.167502 
    ## iter  10 value 15951.008636
    ## iter  20 value 15602.509447
    ## iter  30 value 15320.048597
    ## iter  40 value 15094.406540
    ## iter  50 value 14841.598749
    ## iter  60 value 14593.657450
    ## iter  70 value 14505.696067
    ## iter  80 value 14433.057200
    ## iter  90 value 14278.471377
    ## iter 100 value 14221.582285
    ## final  value 14221.582285 
    ## stopped after 100 iterations
    ## # weights:  63
    ## initial  value 17711.895497 
    ## iter  10 value 16590.252250
    ## iter  20 value 16392.592390
    ## iter  30 value 16303.596223
    ## iter  40 value 16174.482217
    ## iter  50 value 16071.432099
    ## iter  60 value 15910.976829
    ## iter  70 value 15849.429105
    ## iter  80 value 15756.869572
    ## iter  90 value 15638.501744
    ## iter 100 value 15502.201690
    ## final  value 15502.201690 
    ## stopped after 100 iterations
    ## # weights:  179
    ## initial  value 17769.334897 
    ## iter  10 value 16236.683455
    ## iter  20 value 15772.459769
    ## iter  30 value 15639.700027
    ## iter  40 value 15559.574527
    ## iter  50 value 15387.015566
    ## iter  60 value 15230.882001
    ## iter  70 value 15212.902980
    ## iter  80 value 15176.082078
    ## iter  90 value 15118.549139
    ## iter 100 value 15099.041642
    ## final  value 15099.041642 
    ## stopped after 100 iterations
    ## # weights:  295
    ## initial  value 17324.377934 
    ## iter  10 value 15586.659592
    ## iter  20 value 15241.177407
    ## iter  30 value 14828.099076
    ## iter  40 value 14592.808009
    ## iter  50 value 14546.461168
    ## iter  60 value 14525.221534
    ## iter  70 value 14431.982459
    ## iter  80 value 14329.500304
    ## iter  90 value 14177.601619
    ## iter 100 value 14143.184505
    ## final  value 14143.184505 
    ## stopped after 100 iterations
    ## # weights:  63
    ## initial  value 16974.680286 
    ## iter  10 value 16070.209811
    ## iter  20 value 16063.482240
    ## iter  30 value 15922.392874
    ## iter  40 value 15884.442513
    ## iter  50 value 15854.566377
    ## iter  60 value 15843.175579
    ## iter  70 value 15819.887353
    ## iter  80 value 15816.400773
    ## iter  90 value 15815.577234
    ## iter 100 value 15814.662588
    ## final  value 15814.662588 
    ## stopped after 100 iterations
    ## # weights:  179
    ## initial  value 18902.062419 
    ## iter  10 value 16145.122395
    ## iter  20 value 15962.401348
    ## iter  30 value 15697.304938
    ## iter  40 value 15656.269962
    ## iter  50 value 15532.671843
    ## iter  60 value 15282.392799
    ## iter  70 value 15193.681721
    ## iter  80 value 14967.775716
    ## iter  90 value 14530.044261
    ## iter 100 value 14447.201950
    ## final  value 14447.201950 
    ## stopped after 100 iterations
    ## # weights:  295
    ## initial  value 19141.749365 
    ## iter  10 value 15998.295097
    ## iter  20 value 15513.041120
    ## iter  30 value 15333.301940
    ## iter  40 value 15130.542732
    ## iter  50 value 14943.651693
    ## iter  60 value 14821.582256
    ## iter  70 value 14785.835509
    ## iter  80 value 14750.580685
    ## iter  90 value 14726.567902
    ## iter 100 value 14708.853896
    ## final  value 14708.853896 
    ## stopped after 100 iterations
    ## # weights:  63
    ## initial  value 19104.859937 
    ## iter  10 value 16002.764238
    ## iter  20 value 15906.894702
    ## iter  30 value 15808.115989
    ## iter  40 value 15707.777616
    ## iter  50 value 15643.827608
    ## iter  60 value 15613.667227
    ## iter  70 value 15563.245099
    ## iter  80 value 15524.063012
    ## iter  90 value 15444.288920
    ## iter 100 value 15338.005625
    ## final  value 15338.005625 
    ## stopped after 100 iterations
    ## # weights:  179
    ## initial  value 17632.569124 
    ## iter  10 value 16171.082286
    ## iter  20 value 15711.752317
    ## iter  30 value 15513.738832
    ## iter  40 value 15474.640859
    ## iter  50 value 15433.099730
    ## iter  60 value 15404.334963
    ## iter  70 value 15326.553733
    ## iter  80 value 15308.491083
    ## iter  90 value 15303.016654
    ## iter 100 value 15269.618575
    ## final  value 15269.618575 
    ## stopped after 100 iterations
    ## # weights:  295
    ## initial  value 17995.383019 
    ## iter  10 value 15734.609390
    ## iter  20 value 15349.955838
    ## iter  30 value 15144.957545
    ## iter  40 value 15097.352334
    ## iter  50 value 14999.242115
    ## iter  60 value 14823.419730
    ## iter  70 value 14564.994616
    ## iter  80 value 14518.035097
    ## iter  90 value 14493.245387
    ## iter 100 value 14448.248717
    ## final  value 14448.248717 
    ## stopped after 100 iterations
    ## # weights:  63
    ## initial  value 18831.463111 
    ## iter  10 value 16436.463436
    ## iter  20 value 16236.395861
    ## iter  30 value 16178.675139
    ## iter  40 value 16135.164014
    ## iter  50 value 16126.716857
    ## iter  60 value 16092.385513
    ## iter  70 value 16087.876884
    ## iter  80 value 16085.406194
    ## iter  90 value 16083.886861
    ## iter 100 value 16083.175116
    ## final  value 16083.175116 
    ## stopped after 100 iterations
    ## # weights:  179
    ## initial  value 20032.692703 
    ## iter  10 value 16438.801734
    ## iter  20 value 15627.537760
    ## iter  30 value 15423.180043
    ## iter  40 value 15401.795097
    ## iter  50 value 15294.867783
    ## iter  60 value 15175.989547
    ## iter  70 value 15108.142340
    ## iter  80 value 15040.546856
    ## iter  90 value 15028.892581
    ## iter 100 value 15019.395919
    ## final  value 15019.395919 
    ## stopped after 100 iterations
    ## # weights:  295
    ## initial  value 17169.669031 
    ## iter  10 value 15872.792760
    ## iter  20 value 15578.333348
    ## iter  30 value 15312.854054
    ## iter  40 value 15108.055686
    ## iter  50 value 14941.968393
    ## iter  60 value 14877.346727
    ## iter  70 value 14759.738164
    ## iter  80 value 14063.000314
    ## iter  90 value 13961.969857
    ## iter 100 value 13926.139578
    ## final  value 13926.139578 
    ## stopped after 100 iterations
    ## # weights:  63
    ## initial  value 17884.092743 
    ## iter  10 value 16468.953878
    ## iter  20 value 16131.630639
    ## iter  30 value 16107.850154
    ## iter  40 value 16046.361257
    ## iter  50 value 15986.205875
    ## iter  60 value 15880.842924
    ## iter  70 value 15707.694090
    ## iter  80 value 15483.587986
    ## iter  90 value 15417.788276
    ## iter 100 value 15368.610784
    ## final  value 15368.610784 
    ## stopped after 100 iterations
    ## # weights:  179
    ## initial  value 19067.588101 
    ## iter  10 value 15857.864594
    ## iter  20 value 15437.056021
    ## iter  30 value 15310.888288
    ## iter  40 value 15174.214715
    ## iter  50 value 14916.621514
    ## iter  60 value 14763.826487
    ## iter  70 value 14584.967308
    ## iter  80 value 14493.738933
    ## iter  90 value 14448.003134
    ## iter 100 value 14375.968957
    ## final  value 14375.968957 
    ## stopped after 100 iterations
    ## # weights:  295
    ## initial  value 18865.373150 
    ## iter  10 value 15407.283197
    ## iter  20 value 14873.674501
    ## iter  30 value 14524.571740
    ## iter  40 value 14136.609582
    ## iter  50 value 13881.756983
    ## iter  60 value 13719.591044
    ## iter  70 value 13616.842799
    ## iter  80 value 13489.025285
    ## iter  90 value 13344.027099
    ## iter 100 value 13174.414569
    ## final  value 13174.414569 
    ## stopped after 100 iterations
    ## # weights:  63
    ## initial  value 17276.419931 
    ## iter  10 value 16572.526028
    ## iter  20 value 16517.296669
    ## iter  30 value 16505.445596
    ## iter  40 value 16375.848112
    ## iter  50 value 15983.043746
    ## iter  60 value 15484.976943
    ## iter  70 value 15107.503346
    ## iter  80 value 14862.858722
    ## iter  90 value 14687.049414
    ## iter 100 value 14505.401518
    ## final  value 14505.401518 
    ## stopped after 100 iterations
    ## # weights:  179
    ## initial  value 18356.837504 
    ## iter  10 value 16160.861225
    ## iter  20 value 15995.946742
    ## iter  30 value 15956.738833
    ## iter  40 value 15840.150207
    ## iter  50 value 15556.536128
    ## iter  60 value 15540.471899
    ## iter  70 value 15524.539130
    ## iter  80 value 15516.376902
    ## iter  90 value 15510.581964
    ## iter 100 value 15447.165681
    ## final  value 15447.165681 
    ## stopped after 100 iterations
    ## # weights:  295
    ## initial  value 20247.253869 
    ## iter  10 value 15518.727014
    ## iter  20 value 15282.353935
    ## iter  30 value 15239.610717
    ## iter  40 value 15204.529824
    ## iter  50 value 15134.579121
    ## iter  60 value 14990.013231
    ## iter  70 value 14733.485727
    ## iter  80 value 14529.117587
    ## iter  90 value 14313.904588
    ## iter 100 value 14260.870518
    ## final  value 14260.870518 
    ## stopped after 100 iterations
    ## # weights:  63
    ## initial  value 17275.924591 
    ## iter  10 value 16589.435469
    ## iter  20 value 16550.626643
    ## iter  30 value 16490.915458
    ## iter  40 value 16463.803605
    ## iter  50 value 16439.979990
    ## iter  60 value 16415.454237
    ## iter  70 value 16411.732348
    ## iter  80 value 16389.589638
    ## iter  90 value 16363.460540
    ## iter 100 value 16338.278409
    ## final  value 16338.278409 
    ## stopped after 100 iterations
    ## # weights:  179
    ## initial  value 17532.424049 
    ## iter  10 value 15787.857014
    ## iter  20 value 15558.948383
    ## iter  30 value 15236.661891
    ## iter  40 value 15028.809717
    ## iter  50 value 15010.272523
    ## iter  60 value 14971.657736
    ## iter  70 value 14946.666545
    ## iter  80 value 14885.133635
    ## iter  90 value 14844.877041
    ## iter 100 value 14769.716051
    ## final  value 14769.716051 
    ## stopped after 100 iterations
    ## # weights:  295
    ## initial  value 19699.357511 
    ## iter  10 value 16301.091981
    ## iter  20 value 15754.618084
    ## iter  30 value 15608.789110
    ## iter  40 value 15268.991444
    ## iter  50 value 15163.782816
    ## iter  60 value 15019.188892
    ## iter  70 value 14968.086552
    ## iter  80 value 14837.499378
    ## iter  90 value 14734.819335
    ## iter 100 value 14674.855249
    ## final  value 14674.855249 
    ## stopped after 100 iterations
    ## # weights:  295
    ## initial  value 27597.240107 
    ## iter  10 value 24306.142376
    ## iter  20 value 24026.091951
    ## iter  30 value 23556.689220
    ## iter  40 value 22798.625135
    ## iter  50 value 22143.798542
    ## iter  60 value 21994.473931
    ## iter  70 value 21685.497943
    ## iter  80 value 21478.176596
    ## iter  90 value 21367.903275
    ## iter 100 value 21081.812535
    ## final  value 21081.812535 
    ## stopped after 100 iterations

``` r
confusionMatrix(predict(fit_nnet, myTesting), myTesting$class)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   A   B   C   D   E
    ##          A 698  76  75  59  52
    ##          B  24 336  93  52 239
    ##          C   1   7  31   0  31
    ##          D 391 250 421 521 298
    ##          E   2  90  64  11 101
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.43            
    ##                  95% CI : (0.4145, 0.4457)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.287           
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.6254  0.44269 0.045322   0.8103  0.14008
    ## Specificity            0.9067  0.87105 0.987959   0.5854  0.94785
    ## Pos Pred Value         0.7271  0.45161 0.442857   0.2770  0.37687
    ## Neg Pred Value         0.8589  0.86694 0.830522   0.9403  0.83037
    ## Prevalence             0.2845  0.19347 0.174356   0.1639  0.18379
    ## Detection Rate         0.1779  0.08565 0.007902   0.1328  0.02575
    ## Detection Prevalence   0.2447  0.18965 0.017843   0.4795  0.06832
    ## Balanced Accuracy      0.7661  0.65687 0.516640   0.6978  0.54396

Compared to the two other approach, result is very deceiving.

## Classification for test

The random forest algorithm gave so good results that it will be used to
estimate the results on the real Test set.

``` r
predict(fit_rf, testing)
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
