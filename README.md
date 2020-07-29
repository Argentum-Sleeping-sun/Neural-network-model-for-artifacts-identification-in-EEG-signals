# Neural-network-model-for-artifacts-identification-in-EEG-signals
One of the main methods for research of the holistic activity system of human brain is the method of electroencephalography (EEG). The article describes the development of an intelligent neural network model aimed at detecting the artifacts in EEG signals. We have conducted the series of experiments to investigate the performance of different neural networks architectures for the task of artifact detection. As a result, the performance rates for different ML methods were obtained. We have developed (we suggest) the neural network model based on U-net architecture with recurrent networks elements.

The EEG data for our research were provided by Russian Academy of Education Institute. The data was obtained through a 64-channel electroencephalograph from 64 active electrodes placed according to the international «10-10%» system. Data analysis was performed. As a result, the distance between artifacts in the signals and the maximum duration of each type of artifact were defined. An optimal time window for artifact recognition based on the maximum length of the "Blink" artifact was allocated using analytical functions. Since the data was manually filtered from the artifacts and the database was small (9574 samples with artifacts), there was a problem with the quality of training of the neural network. The database was expanded using augmentation method, which partially influenced the learning process (28,722 samples with augmentation).

Files descriptions:

NeuralNetwork.py - The NeuralNetwork.py library allows you to create samples for training a neural network based on arrays that are generated using the Parse_data.py library. The main method is 'prepare_data', which is based on information about artifacts, a database of EEG signals, used channels, and the size of the input window (in seconds) and a given percentage of samples with a normal state.

parse_data.py - The Parse_data.py library is used to convert Markers files into an associative array containing all the artifact information for each record in .edf files.

statistics.py - The statistics.py library contains the methods for analyzing the database
Unet.py - The Unet.py library contains U-net based neural network architecture

main.py - Program file.
