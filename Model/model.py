import numpy as np
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from matplotlib import pyplot
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,accuracy_score,fbeta_score
# fit and evaluate a model

def evaluate_model(
    trainX,
    trainy,
    testX,
    testy,
    ):

    # define model

    (verbose, epochs, batch_size) = (0, 25, 64)
    (n_timesteps, n_features, n_outputs) = (trainX.shape[1],
            trainX.shape[2], trainy.shape[1])

    # reshape data into time steps of sub-sequences

    (n_steps, n_length) = (5, 11)
    print(trainX.shape,testX.shape)
    trainX = trainX.reshape((trainX.shape[0], n_steps, n_length,n_features))
    testX = testX.reshape((testX.shape[0], n_steps, n_length,n_features))

    # define model

    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3,
              activation='relu'), input_shape=(None, n_length,
              n_features)))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3,
              activation='relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(100))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    # fit network

    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size,
              verbose=verbose)
    model.save('activity_recog')
    # evaluate model

    (_, accuracy) = model.evaluate(testX, testy, batch_size=batch_size,
                                   verbose=2)
    
    
    # predict probabilities for test set
    yhat_probs = model.predict(testX, verbose=0)
    #print(yhat_probs)
    # predict crisp classes for test set
    yhat_classes = np.argmax(yhat_probs,axis=1)
    test_classes=[]
    for i in testy:
        test_classes.append(np.where(i==1)[0][0])
    print(yhat_classes)
    
    print(test_classes)
    
    activity_names=['falling_1','falling_2','bending','jumping','running','sitting','standing','walking','getting_up_fast']
    matrix = confusion_matrix(test_classes, yhat_classes)
    print(matrix)
    r=[]
    c=[]
    sum=0
    for i in range(len(matrix)):
        rsum=0
        for j in range(len(matrix[0])):
            rsum+=matrix[i][j]
        sum+=rsum
        r.append(rsum)
    
    for i in range(len(matrix[0])):
        csum=0
        for j in range(len(matrix)):
            csum+=matrix[j][i]
        c.append(csum)
    
    val={}
    for i in range(len(matrix)):
        val[i]={'tp':0,'tn':0,'fp':0,'fn':0}
        val[i]['tp']=matrix[i][i]
        val[i]['tn']=sum-r[i]-c[i]+matrix[i][i]
        val[i]['fp']=r[i]-matrix[i][i]
        val[i]['fn']=c[i]-matrix[i][i]
        val[i]['acc']=(val[i]['tp']+val[i]['tn'])/sum
        val[i]['precision']=val[i]['tp']/(val[i]['tp']+val[i]['fp'])
        val[i]['recall']=val[i]['tp']/(val[i]['tp']+val[i]['fn'])
    print(val)
    for i in val:
        #print(i)
        print(activity_names[i]+": ")
        print("Accuracy: ",end="")
        print(val[i]['acc'])
        print("Precision: ",end="")
        print(val[i]['precision'])
        print("Recall: ",end="")
        print(val[i]['recall'])
        print()
    print("Classification Report:\n", classification_report(test_classes, yhat_classes))
    print("FBeta score:",fbeta_score(test_classes, yhat_classes, average='weighted', beta=0.5))
    accuracy= accuracy_score(test_classes, yhat_classes)
    return accuracy
