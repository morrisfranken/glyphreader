# GlyphReader
A deeplearning approach to classifying the ancient Egyptian hieroglyphs. The source code is written in **python3** using the popular [Keras](https://keras.io/) framework (with a [Tensorflow](https://keras.io/) backend).
It attempts to classify images to their [Gardiner](https://en.wikipedia.org/wiki/Gardiner's_sign_list) labels, such as:

**Image** | ![GitHub Logo](/examples/200000_S29.png) | ![GitHub Logo](/examples/200001_V13.png) | ![GitHub Logo](/examples/200003_G43.png) 
------------ | ------------ | ------------- | -------------
**Gardener Label** | S29 | V13 | G43

In addition to the source code, we also provide a dataset containing 4210 manually annotated images of Egyptian hieroglyphs found in the [Pyramid of Unas](https://en.wikipedia.org/wiki/Pyramid_of_Unas).
The dataset will be automatically downloaded when using `train.py` to train a new classifier, but is also available [here](http://iamai.nl/downloads/GlyphDataset.zip)

## Requirements
- `pip3 install numpy sklearn scipy pyyaml h5py`
- [tensorflow](https://www.tensorflow.org/get_started/os_setup)   (tested with version 1.3.1)
- [keras](https://keras.io/#installation)   (tested with version 2.1.2)

## Usage
```
python3 src/classify.py examples
```
Expected output:
```
Predicting the Hieroglyph type...
image name                ::: top 5 best matching hieroglyphs
200000_S29.png            --> ['S29' 'U33' 'R8' 'F12' 'Y3']
200001_V13.png            --> ['V13' 'N37' 'N18' 'V4' 'N35']
200002_V13.png            --> ['V13' 'V31' 'F22' 'N18' 'D156']
200003_G43.png            --> ['G43' 'G17' 'G21' 'W25' 'G25']
200004_D21.png            --> ['D21' 'V30' 'O50' 'D10' 'N5']
200005_O50.png            --> ['O50' 'N5' 'X6' 'D21' 'V25']
200006_X1.png             --> ['X1' 'N29' 'G1' 'D19' 'G4']
200007_M23.png            --> ['M23' 'G39' 'G25' 'I10' 'Aa26']
200008_G43.png            --> ['G43' 'G39' 'G29' 'G1' 'G4']
200009_S29.png            --> ['S29' 'Y3' 'D34' 'N5' 'W18']
200010_V13.png            --> ['V13' 'D52' 'N18' 'G17' 'F22']
200011_M23.png            --> ['M23' 'F16' 'U1' 'N14' 'M4']
200012_G43.png            --> ['G43' 'G21' 'G39' 'G1' 'G17']
200013_D21.png            --> ['D21' 'T30' 'N5' 'X6' 'U1']
200014_O50.png            --> ['O50' 'X1' 'V31' 'U33' 'U1']
200015_V13.png            --> ['V13' 'F22' 'D36' 'D46' 'V4']
200016_G43.png            --> ['G43' 'G17' 'G5' 'G7' 'G4']
200017_S29.png            --> ['S29' 'M195' 'M17' 'W18' 'M1']
```

## Training
In case you would like to train your own classifier, use `train.py`. It takes no arguments, but when running it for the first time it will download the dataset, and starts training. Training itself consist of 2 phases:

1. **Feature Extraction** extract deeplearning features from the images (corresponding to the `avg_pool` layer from the `InceptionV3` network).
2. **Train Classifier** train an SVM on the deeplearning features
If you do not have a GPU, or simply want to retrain the classifier, it is possible to skip the first step and download the precomputed features directly at [http://iamai.nl/downloads/features.npy](http://iamai.nl/downloads/features.npy), store them in `intermediates/features.npy`.
