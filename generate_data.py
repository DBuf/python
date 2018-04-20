import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('access_logs_201612.txt',sep=" ", header = None)
print ("data loaded")

df = df.drop(df.columns[[1,2,5]], axis=1)
df.rename(columns={0: 'ip',3: 'date',4: 'time',6:'request',7:'rc',8:'size',9:'referer',10:'client',11:'in',12:'out',13:'us'}, inplace=True)

df.insert(loc=3, column='datetime', value=pd.to_datetime(df['date'] + ' ' + df['time']))
df = df.drop('date', axis=1)
df = df.drop('time', axis=1)

df.insert(loc=4, column='fail', value=(df.rc!='rc:200').astype(int))
df = df.drop('rc', axis=1)

df.set_index('datetime', inplace=True)

ts=df.resample('1S').apply({'ip':'nunique','request':'nunique','fail':'sum','size':'count'})

ts=ts.fillna(0)

TS = np.array(ts)
TS = TS[:100000]
print(np.isnan(TS).any())
print(TS.shape)
#TS = TS[0:13,:]
#print(TS.shape)
#print(TS)
num_periods = 60 #number of timestep given as input
f_horizon = 10  #forecast horizon, 10 second into the future

x_data = TS[:((len(TS)-f_horizon)-((len(TS)-f_horizon) % num_periods))]
#x_batches = x_data.reshape(-1, num_periods, 4)
x_batches2= TS[np.arange(num_periods)[None, :] + np.arange(len(TS)-num_periods-f_horizon)[:, None]]

y_data = TS[num_periods:(len(TS)-(len(TS) % num_periods))+f_horizon:num_periods]
#y_batches = y_data.reshape(-1, f_horizon, 4)
y_batches2= TS[num_periods+np.arange(f_horizon)[None, :]+np.arange(len(TS)-num_periods-f_horizon)[:, None]]
y_batches2=np.sum(y_batches2[:,:,3], axis=1)

#print (x_batches2.shape)
#print (x_batches2[0:3])

#print (y_batches2.shape)
#print (y_batches2[0:3])


test_size=5000
x_train=x_batches2[:-test_size]
x_test=x_batches2[-test_size:]
y_train=y_batches2[:-test_size]
y_test=y_batches2[-test_size:]

p1 = np.random.permutation(len(x_train))
x_train_time=x_train[p1]
y_train_time=y_train[p1]
p2 = np.random.permutation(len(x_test))
x_test_time=x_test[p2]
y_test_time=y_test[p2]

#print(x_train.shape)
#print(x_test.shape)
#print(y_train.shape)
#print(y_test.shape)

np.save('x_train_time', x_train_time)
np.save('x_test_time', x_test_time)
np.save('y_train_time', y_train_time)
np.save('y_test_time', y_test_time)


p3 = np.random.permutation(len(x_batches2))
x_batches2=x_batches2[p1]
y_batches2=y_batches2[p1]

x_train=x_batches2[:-test_size]
x_test=x_batches2[-test_size:]
y_train=y_batches2[:-test_size]
y_test=y_batches2[-test_size:]

np.save('x_train', x_train)
np.save('x_test', x_test)
np.save('y_train', y_train)
np.save('y_test', y_test)

mean=np.mean(y_train)
MSE=np.mean(np.square(y_test-mean))
print(mean)
print(MSE)




