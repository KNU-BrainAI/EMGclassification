# EMGhandnet

- Before starting our project, we tested the previous study's model. This is just for testing the previous study's model and dataset, this code may not perfectly work. And it can be different from the original paper because we couldn't find the source code and that was made with TensorFlow.

- We attempted to create the model developed in TensorFlow mentioned in the referenced article.  The model we created achieved a higher accuracy of 93.5% for Ninaprodb1, by incorporating additional preprocessing techniques specific to the datasets mentioned in the article with a learning rate of 1e-3.  As for Ninaprodb4, an accuracy of around 88% was achieved with a learning rate of 1e-3. Their models are different from each other.  This accuracy is slightly lower than the value mentioned in the article.

- For the Ninaprodb1 dataset, separate notebooks were created for each process, and finally, a notebook containing all the processes can be found on this GitHub page. For instance, there are notebooks for the model without any preprocessing, the model with only standardization added, and the model with both standardization and wavelet denoising applied. As for Ninaprodb4, there is a model with all the preprocessing steps mentioned in the article. The code has not yet been completely organized and cleaned; it will be updated soon.
  
- I tried using a batch size of 32 in Naveen's code, just like in my models. With the same model, an accuracy value of 91.9% was achieved for NinaproDB1, and an accuracy value of 87.87% was achieved for NinaproDB4 with a learning rate of 1e-3. You can see the results in NinaproDB1withSameArchitectureNinaproDB4andBatchSize32AsInNaveen.ipynb and NinaproDB4withBatchSize32AsInNaveen.ipynb.
  
- The same model with Naveen Github Code is included in NinaproDB4_SameModelwithNaveenGithub.ipynb and NinaproDB1_SameModelwithNaveenGithub.ipynb notebooks. I got the accuracies for NinaproDB1_85.5% and NinaproDB4_67.8%.

- My model is better than the Naveen code, according to my pre-processing techniques.

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
