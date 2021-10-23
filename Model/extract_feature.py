#~/anaconda3/bin/activate root
import pandas as pd
import numpy as np
from scipy import stats

def extract_feature(filename,sensortype):
    df=pd.read_csv(filename)
    
    x_list=[]
    y_list=[]
    z_list=[]
    window_size=50
    step_size=25
    for i in range(0, df.shape[0] - window_size, step_size):
        xs=df['X'].values[i:i+window_size]
        ys=df['Y'].values[i:i+window_size]
        zs=df['Z'].values[i:i+window_size]

        x_list.append(xs)
        y_list.append(ys)
        z_list.append(zs)
    
    df_final=pd.DataFrame()

    #mean 
    df_final[sensortype+'_x_mean']=pd.Series(x_list).apply(lambda x: x.mean())
    df_final[sensortype+'_y_mean']=pd.Series(y_list).apply(lambda x: x.mean())
    df_final[sensortype+'_z_mean']=pd.Series(z_list).apply(lambda x: x.mean())

    #std
    df_final[sensortype+'_x_std']=pd.Series(x_list).apply(lambda x: x.std())
    df_final[sensortype+'_y_std']=pd.Series(y_list).apply(lambda x: x.std())
    df_final[sensortype+'_z_std']=pd.Series(z_list).apply(lambda x: x.std())

    #mad
    df_final[sensortype+'_x_mad']=pd.Series(x_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    df_final[sensortype+'_y_mad']=pd.Series(y_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    df_final[sensortype+'_z_mad']=pd.Series(z_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))

    #min
    df_final[sensortype+'_x_min']=pd.Series(x_list).apply(lambda x: x.min())
    df_final[sensortype+'_y_min']=pd.Series(y_list).apply(lambda x: x.min())
    df_final[sensortype+'_z_min']=pd.Series(z_list).apply(lambda x: x.min())

    #max
    df_final[sensortype+'_x_max']=pd.Series(x_list).apply(lambda x: x.max())
    df_final[sensortype+'_y_max']=pd.Series(y_list).apply(lambda x: x.max())
    df_final[sensortype+'_z_max']=pd.Series(z_list).apply(lambda x: x.max())

    #sma
    df_final[sensortype+'_sma'] =    pd.Series(x_list).apply(lambda x: np.sum(abs(x)/100)) + pd.Series(y_list).apply(lambda x: np.sum(abs(x)/100)) + pd.Series(z_list).apply(lambda x: np.sum(abs(x)/100))

    #energy
    df_final[sensortype+'_x_energy'] = pd.Series(x_list).apply(lambda x: np.sum(x**2)/100)
    df_final[sensortype+'_y_energy'] = pd.Series(y_list).apply(lambda x: np.sum(x**2)/100)
    df_final[sensortype+'_z_energy'] = pd.Series(z_list).apply(lambda x: np.sum(x**2)/100)

    #IQR
    df_final[sensortype+'_x_IQR'] = pd.Series(x_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    df_final[sensortype+'_y_IQR'] = pd.Series(y_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    df_final[sensortype+'_z_IQR'] = pd.Series(z_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))

    # skewness
    df_final[sensortype+'_x_skewness'] = pd.Series(x_list).apply(lambda x: stats.skew(x))
    df_final[sensortype+'_y_skewness'] = pd.Series(y_list).apply(lambda x: stats.skew(x))
    df_final[sensortype+'_z_skewness'] = pd.Series(z_list).apply(lambda x: stats.skew(x))

    # kurtosis
    df_final[sensortype+'_x_kurtosis'] = pd.Series(x_list).apply(lambda x: stats.kurtosis(x))
    df_final[sensortype+'_y_kurtosis'] = pd.Series(y_list).apply(lambda x: stats.kurtosis(x))
    df_final[sensortype+'_z_kurtosis'] = pd.Series(z_list).apply(lambda x: stats.kurtosis(x))



    #fft
    x_list_fft = pd.Series(x_list).apply(lambda x: np.abs(np.fft.fft(x))[0:50])
    y_list_fft = pd.Series(y_list).apply(lambda x: np.abs(np.fft.fft(x))[0:50])
    z_list_fft = pd.Series(z_list).apply(lambda x: np.abs(np.fft.fft(x))[0:50])

    # FFT mean
    df_final[sensortype+'_x_mean_fft'] = pd.Series(x_list_fft).apply(lambda x: x.mean())
    df_final[sensortype+'_y_mean_fft'] = pd.Series(y_list_fft).apply(lambda x: x.mean())
    df_final[sensortype+'_z_mean_fft'] = pd.Series(z_list_fft).apply(lambda x: x.mean())

    # FFT std dev
    df_final[sensortype+'_x_std_fft'] = pd.Series(x_list_fft).apply(lambda x: x.std())
    df_final[sensortype+'_y_std_fft'] = pd.Series(y_list_fft).apply(lambda x: x.std())
    df_final[sensortype+'_z_std_fft'] = pd.Series(z_list_fft).apply(lambda x: x.std())

    # FFT mad
    df_final[sensortype+'_x_mad_fft']=pd.Series(x_list_fft).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    df_final[sensortype+'_y_mad_fft']=pd.Series(y_list_fft).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    df_final[sensortype+'_z_mad_fft']=pd.Series(z_list_fft).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))

    # FFT min
    df_final[sensortype+'_x_min_fft']=pd.Series(x_list_fft).apply(lambda x: x.min())
    df_final[sensortype+'_y_min_fft']=pd.Series(y_list_fft).apply(lambda x: x.min())
    df_final[sensortype+'_z_min_fft']=pd.Series(z_list_fft).apply(lambda x: x.min())

    # FFT max
    df_final[sensortype+'_x_max_fft']=pd.Series(x_list_fft).apply(lambda x: x.max())
    df_final[sensortype+'_y_max_fft']=pd.Series(y_list_fft).apply(lambda x: x.max())
    df_final[sensortype+'_z_max_fft']=pd.Series(z_list_fft).apply(lambda x: x.max())

    # FFT sma
    df_final[sensortype+'_sma_fft'] = pd.Series(x_list_fft).apply(lambda x: np.sum(abs(x)/100)) + pd.Series(y_list_fft).apply(lambda x: np.sum(abs(x)/100)) + pd.Series(z_list_fft).apply(lambda x: np.sum(abs(x)/100))

    # FFT energy
    df_final[sensortype+'_x_energy_fft'] = pd.Series(x_list_fft).apply(lambda x: np.sum(x**2)/100)
    df_final[sensortype+'_y_energy_fft'] = pd.Series(y_list_fft).apply(lambda x: np.sum(x**2)/100)
    df_final[sensortype+'_z_energy_fft'] = pd.Series(z_list_fft).apply(lambda x: np.sum(x**2)/100)

    # FFT IQR
    df_final[sensortype+'_x_IQR_fft'] = pd.Series(x_list_fft).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    df_final[sensortype+'_y_IQR_fft'] = pd.Series(y_list_fft).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    df_final[sensortype+'_z_IQR_fft'] = pd.Series(z_list_fft).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))

    # FFT skewness
    df_final[sensortype+'_x_skewness_fft'] = pd.Series(x_list_fft).apply(lambda x: stats.skew(x))
    df_final[sensortype+'_y_skewness_fft'] = pd.Series(y_list_fft).apply(lambda x: stats.skew(x))
    df_final[sensortype+'_z_skewness_fft'] = pd.Series(z_list_fft).apply(lambda x: stats.skew(x))

    # FFT kurtosis
    df_final[sensortype+'_x_kurtosis_fft'] = pd.Series(x_list_fft).apply(lambda x: stats.kurtosis(x))
    df_final[sensortype+'_y_kurtosis_fft'] = pd.Series(y_list_fft).apply(lambda x: stats.kurtosis(x))
    df_final[sensortype+'_z_kurtosis_fft'] = pd.Series(z_list_fft).apply(lambda x: stats.kurtosis(x))



    
    return df_final

def main():

    df_final= pd.DataFrame()
    df_acc=extract_feature('../Dataset/Back_Fall/Accelerometer.csv','a')
    df_ang=extract_feature('../Dataset/Back_Fall/Gyroscope.csv','w')
    df_data= df_acc.join(df_ang)
    df_data['label']=[1]*df_data.shape[0]
    df_final=df_data
    print(df_final.shape)

    df_acc=extract_feature('../Dataset/Bending/Accelerometer.csv','a')
    df_ang=extract_feature('../Dataset/Bending/Gyroscope.csv','w')
    df_data= df_acc.join(df_ang)
    df_data['label']=[2]*df_data.shape[0]
    df_final=df_final.append(df_data)
    print(df_final.shape)
    
    df_acc=extract_feature('../Dataset/Front_Fall/Accelerometer.csv','a')
    df_ang=extract_feature('../Dataset/Front_Fall/Gyroscope.csv','w')
    df_data= df_acc.join(df_ang)
    df_data['label']=[3]*df_data.shape[0]
    df_final=df_final.append(df_data)
    print(df_final.shape)

    df_acc=extract_feature('../Dataset/Getting_Up_Fast/Accelerometer.csv','a')
    df_ang=extract_feature('../Dataset/Getting_Up_Fast/Gyroscope.csv','w')
    df_data= df_acc.join(df_ang)
    df_data['label']=[4]*df_data.shape[0]
    df_final=df_final.append(df_data)
    print(df_final.shape)
    df_acc=extract_feature('../Dataset/Jumping/Accelerometer.csv','a')
    df_ang=extract_feature('../Dataset/Jumping/Gyroscope.csv','w')
    df_data= df_acc.join(df_ang)
    df_data['label']=[5]*df_data.shape[0]
    df_final=df_final.append(df_data)
    print(df_final.shape)
    df_acc=extract_feature('../Dataset/Running/Accelerometer.csv','a')
    df_ang=extract_feature('../Dataset/Running/Gyroscope.csv','w')
    df_data= df_acc.join(df_ang)
    df_data['label']=[6]*df_data.shape[0]
    df_final=df_final.append(df_data)
    print(df_final.shape)
    df_acc=extract_feature('../Dataset/Side_Fall/Accelerometer.csv','a')
    df_ang=extract_feature('../Dataset/Side_Fall/Gyroscope.csv','w')
    df_data= df_acc.join(df_ang)
    df_data['label']=[7]*df_data.shape[0]
    df_final=df_final.append(df_data)
    print(df_final.shape)
    df_acc=extract_feature('../Dataset/Sitting/Accelerometer.csv','a')
    df_ang=extract_feature('../Dataset/Sitting/Gyroscope.csv','w')
    df_data= df_acc.join(df_ang)
    df_data['label']=[8]*df_data.shape[0]
    df_final=df_final.append(df_data)
    print(df_final.shape)
    df_acc=extract_feature('../Dataset/Standing/Accelerometer.csv','a')
    df_ang=extract_feature('../Dataset/Standing/Gyroscope.csv','w')
    df_data= df_acc.join(df_ang)
    df_data['label']=[9]*df_data.shape[0]
    df_final=df_final.append(df_data)
    print(df_final.shape)
    df_acc=extract_feature('../Dataset/Walking/Accelerometer.csv','a')
    df_ang=extract_feature('../Dataset/Walking/Gyroscope.csv','w')
    df_data= df_acc.join(df_ang)
    df_data['label']=[10]*df_data.shape[0]
    df_final=df_final.append(df_data)
    print(df_final.shape)
    df_final.to_csv("../Dataset/final.csv")
main()
