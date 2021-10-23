import pandas as pd
import numpy as np
from numpy import dstack
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load a single file as a numpy array

def load_file(filepath):
    dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


# load a list of files and return as a 3d numpy array

def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)

    # stack group so that features are the 3rd dimension

    loaded = dstack(loaded)
    return loaded


# load a dataset group, such as train or test

def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'

    # load all 9 files as a single array

    filenames = list()

    # total acceleration

    filenames += ['total_acc_x_' + group + '.txt', 'total_acc_y_'
                  + group + '.txt', 'total_acc_z_' + group + '.txt']

    # body acceleration

    filenames += ['body_acc_x_' + group + '.txt', 'body_acc_y_' + group
                  + '.txt', 'body_acc_z_' + group + '.txt']

    # body gyroscope

    filenames += ['body_gyro_x_' + group + '.txt', 'body_gyro_y_'
                  + group + '.txt', 'body_gyro_z_' + group + '.txt']

    # load input data

    X = load_group(filenames, filepath)

    # load class output

    y = load_file(prefix + group + '/y_' + group + '.txt')
    return (X, y)


# load the dataset, returns train and test X and y elements

def load_dataset(prefix=''):

    # load all train

    (trainX, trainy) = load_dataset_group('train', prefix
            + 'HARDataset/')
    print (trainX.shape, trainy.shape)

    # load all test

    (testX, testy) = load_dataset_group('test', prefix + 'HARDataset/')
    print (testX.shape, testy.shape)

    # zero-offset class values

    trainy = trainy - 1
    testy = testy - 1

    # one hot encode y

    #trainy = to_categorical(trainy)
    #testy = to_categorical(testy)
    print (trainX.shape, trainy.shape, testX.shape, testy.shape)
    return (trainX, trainy, testX, testy)


def load_sample():
    loaded = list()
    #filenames=['acc.csv','gyro.csv']
    df_acc=pd.read_csv('../Dataset/final.csv')
    df_acc=df_acc.apply(pd.to_numeric)
    df_acc_X=df_acc.iloc[:,:-2]
    df_acc_Y=df_acc.iloc[:,-1]
    

    acc_train_X, acc_test_X, acc_train_Y, acc_test_Y = train_test_split(df_acc_X,df_acc_Y,test_size=0.30,random_state=42)

    print(acc_train_X.shape, acc_train_Y.shape, acc_test_X.shape, acc_test_Y.shape)
    """
    df_gyro=pd.read_csv('gyro.csv')
    df_gyro=df_gyro.apply(pd.to_numeric)
    df_gyro_X=df_gyro.iloc[:,:-2]
    df_gyro_Y=df_acc.iloc[:,-1]
    
    gyro_train_X, gyro_test_X, gyro_train_Y, gyro_test_Y = train_test_split(df_gyro_X,df_gyro_Y,test_size=0.30,random_state=42)

    print(gyro_train_X.shape, gyro_train_Y.shape, gyro_test_X.shape, gyro_test_Y.shape)
    
    #print(acc_train_Y.equal(gyro_train_Y), acc_test_Y.equal(gyro_test_Y))
    """
    trainX = list()
    trainX.append(acc_train_X.values)
    #trainX.append(gyro_train_X.values)
    trainX = dstack(trainX)

    testX = list()
    testX.append(acc_test_X.values)
    #testX.append(gyro_test_X.values)
    testX=dstack(testX)

    trainY = acc_train_Y.values
    testY = acc_test_Y.values
    
    # zero-offset class values

    trainY = trainY - 1
    testY = testY - 1
    print(trainY.shape,testY.shape)
    # one hot encode y

    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    print (trainX.shape, trainY.shape, testX.shape, testY.shape)
    return (trainX, trainY, testX, testY)

def load_sample_hierar():
    print("Inside hierarchical load data")
    df=pd.read_csv('../Dataset/final.csv')
    df=df.apply(pd.to_numeric)
    scaler = StandardScaler()
    df_X=df.iloc[:,:-2].values
    scaler.fit(df_X)
    df_X=scaler.transform(df_X)
    df_Y=df.iloc[:,-1].values
    
    train_X, test_X, train_Y, test_Y = train_test_split(df_X,df_Y,test_size=0.30,random_state=42)
    print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)

    train_Y = train_Y - 1
    test_Y = test_Y - 1
    return (train_X, train_Y, test_X, test_Y)
load_sample()