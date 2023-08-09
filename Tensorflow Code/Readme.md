# EMGhandnet

Before starting our project, we tested previous study's model.   
This is just for testing previous study's model and dataset, this code may not perfectly work.   
And it can be different from original paper because we couldn't find original source code and that made with tensorflow.

We attempted to create the model developed in TensorFlow mentioned in the referenced article. 
The model we created achieved a higher accuracy of 93.5% for Ninaprodb1, by incorporating additional preprocessing techniques specific to the datasets mentioned in the article. 
As for Ninaprodb4, an accuracy of around 87% was achieved.  This accuracy is slightly lower than the value mentioned in the article.

For the Ninaprodb1 dataset, separate notebooks were created for each individual process, and finally, a notebook containing all the processes can be found on this GitHub page. 
For instance, there are notebooks for the model without any preprocessing, the model with only standardization added, and the model with both standardization and wavelet denoising applied. 
As for Ninaprodb4, there is a model with all the preprocessing steps mentioned in the article. The code has not been completely organized and cleaned yet; it will be updated soon.

## Environment
Python
Tensorflow
Pandas
Numpy

## Reference
[EMGHandNet: A hybrid CNN and Bi-LSTM architecture for hand activity classification using surface EMG signals]   
https://www.sciencedirect.com/science/article/pii/S0208521622000080   
[NinaPro datasets]   
http://ninaweb.hevs.ch/
