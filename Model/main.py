from model import evaluate_model
from load_data import load_sample,load_sample_hierar
import pandas as pd
from hierarchical_model import hierarchical_model_classify
def main():
    #df=pd.read_csv('../Dataset/total.csv')
    
    #print(trainX.shape,trainY.shape,testX.shape,testY.shape)
    """trainX, trainY, testX, testY = load_sample()
    print("LSTM Classification Report")
    acc=evaluate_model(trainX,trainY,testX,testY)
    print("Accuracy:",acc)"""

    trainX, trainY, testX, testY = load_sample_hierar()
    print("Hierarchical classification")
    acc=hierarchical_model_classify(trainX, testX, trainY, testY)
    print("Accuracy:",acc)

main()