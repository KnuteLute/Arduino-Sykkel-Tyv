import sys
import argparse

if __name__ == '__main__':

    args = sys.argv

    if len(args) != 2:
        sys.exit('please provide a path to dataset as command line argument, example: \"python train_model.py dataset\"')
    else:        
        path = sys.argv[1]

    datasetfiles = load_assets(path)



    parser = argparse.ArgumentParser(description='this script train a model for recognising the faces of the legendary Ask, Knut, Sigurd and Jet')
    parser.add_argument('-d', '--data', required=True, help='provide file path to training datset containing folders Ask, Knut, Sigurd and Jet, containing corresponding images')

    args = parser.parse_args()

    model = cv2.dnn.readNetFromTorch('nn4.small2.v1.t7')
