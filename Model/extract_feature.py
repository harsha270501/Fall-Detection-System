#~/anaconda3/bin/activate root
import pandas as pd
import numpy as np
from scipy import stats

def extract_feature(filename,sensortype,label):
    df=pd.read_csv(filename)
    df=df.iloc[:1000]
    print(df.shape)
    x_list=[]
    y_list=[]
    z_list=[]
    window_size=50
    step_size=25
    for i in range(0, df.shape[0] - window_size, step_size):
        if(sensortype=='a'):
            xs=df['ax (m/s^2)'].values[i:i+window_size]
            ys=df['ay (m/s^2)'].values[i:i+window_size]
            zs=df['az (m/s^2)'].values[i:i+window_size]
        else:
            xs=df['wx (rad/s)'].values[i:i+window_size]
            ys=df['wy (rad/s)'].values[i:i+window_size]
            zs=df['wz (rad/s)'].values[i:i+window_size]

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



    df_final['label']=[label]*df_final.shape[0]
    return df_final

def main():
    df_acc=pd.DataFrame()
    df_acc=extract_feature('../Dataset/falling_1/falling_1_acc.csv','a',1)
    df_acc=df_acc.append(extract_feature('../Dataset/falling_2/falling_2_acc.csv','a',2))
    df_acc=df_acc.append(extract_feature('../Dataset/bending/bending_acc.csv','a',3))
    df_acc=df_acc.append(extract_feature('../Dataset/jumping/jumping_acc.csv','a',4))
    df_acc=df_acc.append(extract_feature('../Dataset/running/running_acc.csv','a',5))
    df_acc=df_acc.append(extract_feature('../Dataset/sitting/sitting_acc.csv','a',6))
    df_acc=df_acc.append(extract_feature('../Dataset/standing/standing_acc.csv','a',7))
    df_acc=df_acc.append(extract_feature('../Dataset/walking/walking_acc.csv','a',8))
    
    df_ang=pd.DataFrame()
    df_ang=extract_feature('../Dataset/falling_1/falling_1_gyro.csv','w',1)
    df_ang=df_ang.append(extract_feature('../Dataset/falling_2/falling_2_gyro.csv','w',2))
    df_ang=df_ang.append(extract_feature('../Dataset/bending/bending_gyro.csv','w',3))
    df_ang=df_ang.append(extract_feature('../Dataset/jumping/jumping_gyro.csv','w',4))
    df_ang=df_ang.append(extract_feature('../Dataset/running/running_gyro.csv','w',5))
    df_ang=df_ang.append(extract_feature('../Dataset/sitting/sitting_gyro.csv','w',6))
    df_ang=df_ang.append(extract_feature('../Dataset/standing/standing_gyro.csv','w',7))
    df_ang=df_ang.append(extract_feature('../Dataset/walking/walking_gyro.csv','w',8))

    df_acc.to_csv('acc.csv',index=False)
    df_acc.head()
    df_ang.to_csv('gyro.csv',index=False)
    df_ang.head()

main()
