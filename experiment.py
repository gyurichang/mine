import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from pandas import DataFrame
import numpy as np
import re
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, GRU, Activation, LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.optimizers import Adam
import keras.backend as K
import sklearn.preprocessing
import sklearn
from sklearn.utils.validation import _check_sample_weight
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report


label_binarizer = sklearn.preprocessing.LabelBinarizer()
model = load_model("C:/Users/gyuri/ML_PATH/논문 파일/model/랭킹모델4/GRU-09-0.787990.hdf5")
lookback = 100



def categorize(time):
    new_series = []
    for i in time:
        if re.match('[0-9]*:[0-9]*:[0-9]', i):   # hh: mm: ss형식과 일치하는 경우에만 카테고리화 하기 위함.
            if i == '24:00:00':
                num = 0
            else:
                hr, mi, se = map(int, i.split(':'))   # 리스트 내 모든 원소들에 int형 적용. : 기준으로 split함.
                num = hr * 6 + mi // 10   # 10분 단위로 넘버링이 1씩 증가 하니까 분 나누기 10의 몫을 더해줌.
        else:
            num = -1
        new_series += [num]
    df['categorizedTime'] = new_series
    #label_binarizer.fit(range(max(df['categorizedTime']) + 1))
    #one_hot_time = label_binarizer.transform(df['categorizedTime'])
    del df['SubmittedTime']


def categorize_time(time):
    if re.match('[0-9]*:[0-9]*:[0-9]', time):  # hh: mm: ss형식과 일치하는 경우에만 카테고리화 하기 위함.
        if time == '24:00:00':
            num = 0
        else:
            hr, mi, se = map(int, time.split(':'))  # 리스트 내 모든 원소들에 int형 적용. : 기준으로 split함.
            num = hr * 6 + mi // 10  # 10분 단위로 넘버링이 1씩 증가 하니까 분 나누기 10의 몫을 더해줌.
    else:
        num = -1
    return num

def determineRank2(t, target, amount, k):
    #    t = str(input())   # 시간
    #    n = int(input())   # 현재 랭킹
    #    bid_t = int(input())   # 비딩 가격
    #    w = int(input())   # 요일
    #    h = int(input())   #  주말 여부
    #    k = str(input())   # 키워드

    new_list = []
    encode = LabelEncoder()
    # scaler = MinMaxScaler()
    # x = np.concatenate((t,n,bid_t,w,h,k),axis = 1).reshape(1,6,1)
    t = categorize_time(t)
    bid_t = amount
    y = (bid_t - 70) / (100000 - 70)

    a = {'가려움증': 0, '건선': 1, '건선치료': 2, '건선치료법': 3, '건선한의원': 4, '다한증': 5, '두드러기': 6, '두드러기치료': 7, '두드러기한의원': 8,
         '백반증': 9, '사타구니습진': 10, '성인아토피': 11, '소아아토피': 12, '습진': 13, '습진치료': 14, '습진한의원': 15, '아토피': 16, '아토피치료': 17,
         '아토피한의원': 18,
         '안면홍조': 19, '알레르기': 20, '장미색비강진': 21, '접촉성피부염': 22, '지루성두피염': 23, '지루성피부염': 24, '지루성피부염치료': 25,
         '지루성피부염한의원': 26, '한포진': 27,
         '한포진치료': 28, '한포진한의원': 29}

    new_list = [t, target, y, a[k]]
    new_list = np.array(new_list)
    new_list = new_list.reshape(1, 1, 4)  # batch_size, timesteps, input_dim
    rank = model.predict(new_list)
    rank = rank.item(0)
    rank = round(rank)
    return rank

"""
def determinehamsu(t, n, bid_t, w, h, k):
    encode = LabelEncoder()
    rank = determineRank(t, n, bid_t, w, h, k)
    df = pd.DataFrame(
        data={'Time': [t], 'Now_Rank': [n], 'Amount': [bid_t], 'Weekday': [w], 'Holiday': [h], 'Keyword': [k]},
        columns=['Time', 'Now_Rank', 'Amount', 'Weekday', 'Holiday', 'Keyword'])

    if bid_t <= 6010:

        for bid_t in range(bid_t, 6010, 10):
            rank = determineRank(t, n, bid_t, w, h, k)

            df = df.append({'Time': t,
                            'Now_Rank': n,
                            'Amount': bid_t,
                            'Weekday': w,
                            'Holiday': h,
                            'Keyword': k,
                            'Target': rank}, ignore_index=True)
            df.to_csv("new_sample5.csv")
            print(rank)
            print(bid_t)
        return df

    elif bid_t > 6010:
        for bid_t in range(bid_t, 0, -10):
            rank = determineRank(t, n, bid_t, w, h, k)
            df = df.append({'Time': t,
                            'Now_Rank': n,
                            'Amount': bid_t,
                            'Weekday': w,
                            'Holiday': h,
                            'Keyword': k,
                            'Target': rank}, ignore_index=True)
            df.to_csv("sample6.csv")
            print(rank)
            print(bid_t)
        return df.item(0)
"""





def determinehamsu2(time, target, amount, k):
    encode = LabelEncoder()
    rank = determineRank2(time, amount, target, k)
    df = []
    df = pd.DataFrame(
        data={'Time': [time], 'Rank': [target], 'Amount': [amount], 'Keyword': [k]},
        columns=['Time', 'Amount', 'Rank', 'Keyword'])

    if amount <= 100000:

        for amount in range(amount, 100010, 10):
            rank = determineRank2(time, amount, target, k)
            df = []
            df = df.append({'Time': time,
                            'Amount': amount,
                            'Rank' : target,
                            'Keyword': k,
                            'Now_rank': rank}, ignore_index=True)
            df.to_csv("new_sample7.csv")
            print(rank)
            print(amount)
        return df

    elif amount > 6010:
        for bid_t in range(amount, 0, -10):
            rank = determineRank2(time, amount, target, k)
            df = df.append({'Time': time,
                            'Amount': amount,
                            'Rank' : target,
                            'Keyword': k,
                            'Target': rank}, ignore_index=True)
            df.to_csv("sample6.csv")
            print(rank)
            print(bid_t)
        return df.item(0)

def getscore(keyword):
   new_list = []
   a = {'가려움증': 0, '건선': 1, '건선치료': 2, '건선치료법': 3, '건선한의원': 4, '다한증': 5, '두드러기': 6, '두드러기치료': 7, '두드러기한의원': 8,
        '백반증': 9, '사타구니습진': 10, '성인아토피': 11, '소아아토피': 12, '습진': 13, '습진치료': 14, '습진한의원': 15, '아토피': 16, '아토피치료': 17,
        '아토피한의원': 18, '안면홍조': 19, '알레르기': 20, '장미색비강진': 21, '접촉성피부염': 22, '지루성두피염': 23, '지루성피부염': 24, '지루성피부염치료': 25,
        '지루성피부염한의원': 26, '한포진': 27, '한포진치료': 28, '한포진한의원': 29}


def create_dataset(dataset, lookback):
  dataX, dataY = [], []
  for i in range(len(dataset) - lookback):
    a = dataset[i:(i+ lookback)]
    dataX.append(a)
    dataY.append(dataset[i+lookback])
  return np.array(dataX), np.array(dataY)


def normalization(x):
    x_np = np.asarray(x)
    return (x_np - x_np.min()) / (x_np.max() - x_np.min())

filename = 'C:/Users/gyuri/Desktop/진짜수정완료.csv'
df = pd.read_csv('C:/Users/gyuri/Desktop/진짜수정완료.csv', encoding='euc-kr', delimiter=',')
df = pd.DataFrame(df)
df.info()
del df['ExecutedDate']
del df['queueName']
del df['cycleMinutes']
del df['shoudBid']
del df['fetchMethod']
del df['siteUrl']
del df['retryQueueName']
del df['retryCount']
del df['Submitted']
del df['submittedAt']
del df['SubmittedDate']
del df['nccKeywordId']
del df['statDt']
del df['statDtTimestamp']
del df['delaySeconds']
del df['customerId']
del df['bidStatusCode']
del df['Holiday']
del df['Weekday']
df.groupby('currentRank').count()
df.groupby('targetRank').count()
print(df['SubmittedTime'])
print(df['keyword'])
df.groupby('bidAmt').count()
scaler = MinMaxScaler()
encoder = LabelEncoder()


normalization(df['bidAmt'])

categorize(df['SubmittedTime'])
#label_binarizer = sklearn.preprocessing.LabelBinarizer()
#label_binarizer.fit(range(max(df['categorizedTime'])+1))
#one_hot_time = label_binarizer.transform(df['categorizedTime'])
#print('{0}'.format(one_hot_time))

#one_hot_time.shape

print(df['categorizedTime'])
#del df['SubmittedTime']
df['keyword'] = encoder.fit_transform(df['keyword'])
print(df['keyword'])
df.info()
#df['Holiday'].loc[df.Holiday=='Y']=1
#df['Holiday'].loc[df.Holiday == 'N']=0

df.info()
seed = tf.set_random_seed(42)

first_file_info = df.values[0:120000].astype(np.int)
print("file info.shape : ", first_file_info.shape)
print("file info[0] : ", first_file_info[0])


time =  first_file_info[:,-1:]
print("date.shape : ", time.shape)
print("date[0] : ", time[0])
print("=" * 120)

now = first_file_info[:,:1]
print("now.shape : ", now.shape)
print("now[0] : ", now[0])
print("="*120)

target = first_file_info[:,1:2]
print("target.shape : ", target.shape)
print("target[0] : ", target[0])
print("="*120)

amount = first_file_info[:, 2:3]
#norm_amount = scaler.fit_transform(amount)
print("amount.shape : ", amount.shape)
print("amount[0] : ", amount[0])
#print("norm_amount[0]: ", norm_amount[0])
print("="*120)



keyword = first_file_info[:, 3:4]
print("keyword.shape : ", keyword.shape)
print("keyword[0]: ", keyword[0])
print("="*120)

x = np.concatenate((time, amount, target, keyword), axis = 1)
print("x.shape : ", x.shape)
print("x[0] : ", x[0])
print("x[-1] : ", x[-1])
print("="*100)

y = now
print("y[0] : ", y[0])
print("y[-1] : ", y[-1])


"""
train_size = int(len(x)*0.7)
test_size = len(x)-train_size
train, test = x[0:train_size], x[train_size:len(x)]
print(len(train), len(test))

X_train, Y_train= create_dataset(train, 100)
X_test, Y_test = create_dataset(test, 100)

Y_train = y[0:83900,]
Y_test = y[83900:119800,]

X_train.shape
X_test.shape
"""


sm = SMOTE(sampling_strategy='auto', random_state=seed)

X_train, X_test,Y_train , Y_test = train_test_split(x,y, test_size = 0.3)


X_resampled, Y_resampled = sm.fit_sample(X_train,Y_train)

X_train.shape
X_test.shape

x.shape
x = x.reshape(120000,1,4)
X_train = X_train.reshape(84000, 1, 4)
X_test = X_test.reshape(36000, 1, 4)

modelpath = "C:/Users/gyuri/ML_PATH/논문 파일/model/랭킹모델4/LSTM10PM-{epoch:02d}-{val_loss:4f}.hdf5"
early_stopping_callback = EarlyStopping(monitor = 'acc', patience = 10)
checkpointer = ModelCheckpoint(filepath = modelpath, monitor = 'acc', verbose = 1, save_best_only = True)


y_pred = model.predict(X_test)

print(y_pred)

model= Sequential()
model.add(LSTM(64, input_shape = (1,4)))
model.add(LSTM(32, activation='relu'))
model.add(LSTM(16, activation = 'relu'))
model.add(LSTM(8, activation='relu'))
model.add(LSTM(4, activation = 'relu'))
model.add(Dense(1))


model.summary()



history = model.fit(X_train, Y_train, epochs = 30, batch_size=1, verbose = 2, shuffle=False, callbacks=[early_stopping_callback, checkpointer],
          validation_data=(X_test,Y_test))

k = 4
num_epochs = 30

skf = StratifiedKFold(n_splits=k, shuffle=False, random_state=seed)

for train, test in skf.split(x,y):
    model = Sequential()
    model.add(GRU(32, batch_input_shape= (1,1,4), return_sequences=True, stateful=True))
    model.add(GRU(16, return_sequences=True, stateful=True))
    model.add(GRU(8, activation = 'relu', return_sequences = True, stateful = True))
    model.add(GRU(4, activation ='relu', stateful = True))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mae', metrics=['acc'])
    model.fit(x[train], y[train], epochs=num_epochs, batch_size=100, verbose=0, callbacks=[early_stopping_callback, checkpointer],
               validation_data = (x[test], y[test]))
    model.summary()


k = 4
num_epochs = 30

skf = StratifiedKFold(n_splits=k, shuffle=False, random_state=seed)

#for train, test in skf.split(x,y):

model2 = Sequential()
model2.add(LSTM(32, input_shape= (1,4)))
model2.add(LSTM(16, return_sequences = True, return_state = True))
model2.add(LSTM(8 ,return_sequences = True, return_state = True))
model2.add(LSTM(4))
model2.add(Dense(1))
model2.compile(optimizer='adam', loss='mae', metrics=['acc'])
model2.fit(X_train, Y_train, epochs=num_epochs, batch_size=1, verbose=0, callbacks=[early_stopping_callback, checkpointer],
           validation_data = (X_test, Y_test))


k = 4
num_epochs = 30

skf = StratifiedKFold(n_splits=k, shuffle=False, random_state=seed)

for train, test in skf.split(x,y):
    model2 = Sequential()
    model2.add(GRU(64, input_shape= (1,4)))
    model2.add(Dense(32))
    model2.add(Dense(16))
    model2.add(Dense(8))
    model2.add(Dense(4))
    model2.add(Dense(1))
    model2.compile(optimizer='adam', loss='mae', metrics=['acc'])
    model2.fit(x[train], y[train], epochs=num_epochs, batch_size=1, verbose=0, callbacks=[early_stopping_callback, checkpointer],
               validation_data = (x[test], y[test]))

    model2.summary()


model = Sequential()
model.add(LSTM(32, batch_input_shape = (1,1,4), return_sequences=True, stateful=True))
model.add(LSTM(16, return_sequences=True, stateful=True))
model.add(LSTM(8, activation = 'relu', return_sequences = True, stateful = True))
model.add(LSTM(4, activation ='relu', stateful = True, return_sequences=True))
model.add(LSTM(2,  stateful=True))
model.add(Dense(1))
model.compile(optimizer=Adam(lr=0.01), loss='mae', metrics=['acc'])
model.fit(X_train, Y_train, epochs=30, batch_size=1, verbose=0, callbacks=[early_stopping_callback, checkpointer],
           validation_data = (X_test, Y_test))


model = Sequential()
model.add(LSTM(64, input_shape=(1, 4)))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

model.summary()



def determineRank2(t, target, amount, k):
    #    t = str(input())   # 시간
    #    n = int(input())   # 현재 랭킹
    #    bid_t = int(input())   # 비딩 가격
    #    w = int(input())   # 요일
    #    h = int(input())   #  주말 여부
    #    k = str(input())   # 키워드

    new_list = []
    encode = LabelEncoder()
    # scaler = MinMaxScaler()
    # x = np.concatenate((t,n,bid_t,w,h,k),axis = 1).reshape(1,6,1)
    t = categorize_time(t)
    bid_t = amount
    y = (bid_t - 70) / (100000 - 70)

    a = {'가려움증': 0, '건선': 1, '건선치료': 2, '건선치료법': 3, '건선한의원': 4, '다한증': 5, '두드러기': 6, '두드러기치료': 7, '두드러기한의원': 8,
         '백반증': 9, '사타구니습진': 10, '성인아토피': 11, '소아아토피': 12, '습진': 13, '습진치료': 14, '습진한의원': 15, '아토피': 16, '아토피치료': 17,
         '아토피한의원': 18,
         '안면홍조': 19, '알레르기': 20, '장미색비강진': 21, '접촉성피부염': 22, '지루성두피염': 23, '지루성피부염': 24, '지루성피부염치료': 25,
         '지루성피부염한의원': 26, '한포진': 27,
         '한포진치료': 28, '한포진한의원': 29}

    new_list = [t, target, y, a[k]]
    new_list = np.array(new_list)
    new_list = new_list.reshape(1, 1, 4)
    rank = model.predict(new_list)
    rank = rank.item(0)
    rank = round(rank)
    return rank

determineRank2("21:30:04", 3620, 2, "습진한의원")
determineRank2("15:30:04", 5100, 2, "한포진치료")


model2 = Sequential()
model2.add(LSTM(64, input_shape = (1,4)))
model2.add(Dense(1))

model2.compile(optimizer=Adam(lr = 0.01), loss = 'mae', metrics=['acc'])

model2.fit(X2_train, Y2_train, epochs = 30, batch_size=100, verbose = 2, shuffle=False, callbacks=[early_stopping_callback, checkpointer],
          validation_data=(X2_test,Y2_test))

model = load_model('C:/Users/gyuri/ML_PATH/논문 파일/model/랭킹모델4/12-0.653181.hdf5')
predict_X2_train = model.predict(X2_train)
predict_X2_test = model.predict(X2_test)

predict_X2_train.shape
predict_X2_test.shape




# x = np.concatenate((time, amount, target, keyword), axis = 1)
# new_list = [t, amount, target, k]

model = load_model('C:/Users/gyuri/ML_PATH/논문 파일/model/랭킹모델4/LSTM10PM-02-0.780523.hdf5')

determineRank2("21:30:04", 2, 3620, "습진한의원") #GRU 모델, LSTM -맞음, 1위
determineRank2("21:40:03", 2, 3370, "습진치료")   #GRU 모델 -틀림, LSTM - 맞음, 2위
determineRank2("4:00:04", 2, 5360, "습진치료") #GRU 모델 - 틀림, LSTM - 틀림 1위
determineRank2("1:00:04", 2, 5160, "한포진치료")  # GRU 모델 - 맞음, LSTM - 맞음, 2위
determineRank2("18:00:04", 2, 16670, "소아아토피") # GRU 모델 -  맞음, LSTM - 틀림, 1위


determineRank2("13:40:14", 2, 12000,"안면홍조")  # GRU 모델 - 틀림, LSTM - 틀림,20위
determineRank2("22:00:14", 2, 20830, "두드러기한의원") # GRU 모델 - 틀림, 3위
determineRank2("16:00:12", 2, 20130, "아토피한의원") # GRU 모델 - 틀림, 2위
determineRank2("17:20:12", 2, 6660, "습진한의원") #GRU 모델 - 맞음, LSTM - 틀림, 1위
determineRank2("00:00:04", 5, 1710, "접촉성피부염") # GRU 모델 - 맞음, 2위
determineRank2("23:00:04", 1, 17860, "아토피치료")  # GRU 모델 - 맞음, 2위
determineRank2("23:00:04", 1, 6000, "아토피치료")


determinehamsu2("0:00:04", 2, 10, "두드러기")

determinehamsu2("18:10:20", 2, 10, "아토피")
