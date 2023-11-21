# Arduino-Sykkel-Tyv
Catching the fucking bike thief

To train the model SVM, do `python SVM.py`.
To train the model KNN, do `python model_maker.py`.

To run facial detection on Ask, Knut, Sigurd and Jet:
- install libraries, virtual environment recommended
- do `python video.py`

To train the facial detection model:
- install libraries, virtual environment recommended
- do `python train.py --path <path to dataset>` and replace `<path to dataset>` with a dataset folder
- the dataset folder has to contain folders: Ask, Knut, Jet, Sigurd with corresponding images.

# pre-trained model
We are using the pre-trained model [nn4.small2.v1.t7](https://cmusatyalab.github.io/openface/models-and-accuracies/).

# read further
- [pyimagesearch.com](https://pyimagesearch.com/2018/09/24/opencv-face-recognition/)
