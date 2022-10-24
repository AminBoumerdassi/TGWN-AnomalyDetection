# TGWN-AnomalyDetection (read "tee-gwin")
### Transient Gravitational Wave Noise Anomaly Detection
A convolutional autoencoder trained on transient gravitational wave noise (glitches) from the Gravity Spy database.
Its purpose is for anomaly detection (non-anomaly=glitches, anomaly=astrophysical transient events).
If successful, this will form the glitch rejection mechanism for the GW-ML burst detection pipeline MLy.

Paper on the MLy pipeline: https://arxiv.org/abs/2009.14611 

Paper on the Gravity Spy database: https://iopscience.iop.org/article/10.1088/1361-6382/aa5cea


#To train the autoencoder
1. Download or query the Gravity Spy database to retrieve all of the glitchs' metadata
2. Feed these metadata into MLy's generator function to create timeseries of glitches with real noise
3. Process the glitch labels into a one-hot encoding
4. Train the autoencoder a given glitch type 
