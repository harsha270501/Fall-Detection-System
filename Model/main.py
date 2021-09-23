from model import evaluate_model
from load_data import load_sample

def main():
    trainX, trainY, testX, testY = load_sample()
    acc=evaluate_model(trainX, trainY, testX, testY)
    print(acc)

main()