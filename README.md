# feature-selection-for-unsupervised-learning

Visualizing dataset:<br>
In order to achieve that you should consider the following:<br>

    Use PCA to visualize the datasets using the resulting top k principal components, where k is the desired lower dimensionality for visualization. You can select k=2 or k=3 what is better for the dataset under investigation. <br>
        For 2D visualization, set k=2 to project the data onto a plane.<br>
        For 3D visualization, set k=3 to project the data onto a three-dimensional space.<br>
    Use k-means to<br>
        Cluster samples; set k value to be equal to the number of true classes.<br>
        Cluster features; set k value to be 30% of the number of features.<br>
<br>
Evaluating and Reporting:<br>
Compare the performance of the k-means with the selected features to a baseline k-means that uses all features.<br>

    Run k-means clustering with selected features.<br>
    Run k- means clustering with all features.<br>

 For each run in (1) and (2) you will:<br>

    Set k = number of true clusters in the dataset<br>
    Run k-means 10 times.<br>
    Measure the quality and efficiency of the clustering using AC and NMI:<br>
        Accuracy (AC)<br>
        Normalized Mutual Information (NMI)<br>
        Time (in seconds)<br>
    Report the quality and efficiency of the clustering in each run.<br>
    Report the quality and efficiency of the clustering over all runs (min, max, avg, standard deviation).  <br>


    Hybrid UFS to improve UFS model:<br>
<br>
Combine the two UFS models to improve the final select UFS. Designe, implement, and discuss the performance of your hybrid UFS solution.<br>
Many approaches exist, some common approaches include:<br>

    Ensemble of Methods:<br>
        Apply different unsupervised feature selection methods independently.<br>
        Combine the results, either by voting (majority or weighted), or using some other criterion.<br>
    Sequential Feature Selection:<br>
        Apply unsupervised feature selection methods sequentially, selecting a subset of features at each step.<br>
        Combine the subsets obtained at different steps.

