from typing import Union, Any

import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as md
import warnings
import scipy as sp
import sklearn.preprocessing


#1. Read the CSV file and extract the measured values
#2. Put the measured values into two NumPy Arrays, such as baroA (Sensor-A) and baroB (Sensor-B)
#IMPORTANT: I have edited the CSV file such that it can seperately show the value of Sensor A and Sensor B

baroA, baroB =np.loadtxt('Barometer.csv', delimiter=",", skiprows=1, usecols=(0,1), unpack=True)

print(baroA)
print(baroB)


#3. Convert the measured air pressure (hPa) into the altitude (meters) the altitude increases by 8m per 1 hPa of reduced air pressure

reduced_air_pressure_A = 1013.25 - baroA
altA = reduced_air_pressure_A * 8

'''
print(altA)
range1=min(altA)
range2=max(altA)
print(range1)
print(range2)
'''

reduced_air_pressure_B = 1013.25 - baroB
altB = reduced_air_pressure_B * 8
#print(altB)

'''
print(altB)
range1=min(altB)
range2=max(altB)
print(range1)
print(range2)
'''

#4. Create an additional NumPy Array for the timestamps (start time: 12:30:00)

'''
frequency=50.0
period = 1.0/frequency
'''
start_time = datetime.datetime.strptime("2022-02-01 12:30:00", "%Y-%m-%d %H:%M:%S")
print(start_time)
timestamps = [start_time + datetime.timedelta(seconds=x*1) for x in range(len(altA))]
print(timestamps)

timestamp= []
for i in timestamps:
    z = int(i.strftime("%S"))
    timestamp.append(z)
time_Length=sklearn.preprocessing.scale(timestamp)
'''
offset_start_time = datetime.datetime.strptime("2022-02-01 12:30:00", "%Y-%m-%d %H:%M:%S")
offset_stop_time = datetime.datetime.strptime("2022-02-01 12:30:50", "%Y-%m-%d %H:%M:%S")
'''

#5. Smooth the measured altitude of sensor-A (altA)

Cubic = np.polyfit(timestamp,altA, 3)
print('cubic',Cubic)
# xlist = np.arange(170.0, 350.0, 3.6)
# print('x',xlist)
filtA = np.polyval(Cubic,timestamp)
print(filtA)


''' ALTERNATE
A_list = np.arange(175.0, 350.0, 3.5)
altA_list = A_list[::-1]
matrix_A = np.vstack((np.ones(len(time_length)), time_length, time_length**2, time_length**3)).T

# unknows [a, b, c, d], polyfit
u = np.linalg.inv(matrix_A.T.dot(matrix_A)).dot(matrix_A.T).dot(altA)
#print(u)
# range for x values, polyval
filtA = [u[0] + u[1] * item + u[2] * item**2 + u[3] * item**3 for item in altA] # a + b * x + c * x**2
#print(new_A)

plt.close('all')
plt.plot(altA_list, filtA,'-r',label="cubic regression")
plt.legend()
plt.show()
'''

#6. Smooth the measured altitude of sensor-B (altB) using the moving average filter

def moving_average(x, window_len=4, window='flat'):
    if x.ndim != 1:
        raise ValueError("oth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[int(window_len / 2 - 1):-int(window_len / 2)]


filtB= moving_average(altB)
print(filtB)
print(len(filtB))

''' ALTERNATE
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 50))
    return (cumsum[N:] - cumsum[:-N]) / float(N)
moving_average= running_mean(altB, 4)
print(moving_average)

filtB = np.convolve(moving_average)
print(filtB)
plt.close('all')
plt.plot(filtB,'-r',label="cubic regression")
plt.legend()
plt.show()
'''

''' ALTERNATE
def moving_average(altB, offset=1):
    
    if offset ==0:
        return altB
    if offset <0 or offset > altB.size:
        raise Exception("invalid offset !")

    filtB = [np.nan] * altB.size
    for i in range(i, altB.size):
        if i + offset <= x.size - 1:
            filtB[i] = (altB[i-1] + altB[i] + altB[i+1])/3
            print(filtB)
'''


''' ALTERNATE
filtA = np.poly1d(np.polyfit(altA, altB))
myline = np.linspace(1, 22, 100)
plt.plot(myline, filtA(myline))
plt.show()

filtB = altB.smooth( window_len=4, window='hanning')
print(filtB)
'''

# 7. Calculate the altitude differences resulting from both sensors (filtA - filtB)

diffAB = []
for index,value in zip(filtA,filtB):
        diffAB.append(abs(index-value))
print(diffAB)


# time window boundaries
#offset_start_time = datetime.datetime.strptime("2022-02-01 12:30:00", "%Y-%m-%d %H:%M:%S")
#offset_stop_time = datetime.datetime.strptime("2022-02-01 12:30:50", "%Y-%m-%d %H:%M:%S")

#start_index = [i for i, t in enumerate(timestamps) if (t - offset_start_time).total_seconds() > 0.0][0]
#stop_index = [i for i, t in enumerate(timestamps) if (t - offset_stop_time).total_seconds() > 0.0][0]

#print(start_index,stop_index)


plt.figure("altA", figsize=(20,8))
plt.title(" Sensor-A (" + start_time.date().strftime("%Y-%m-%d") + ")")
plt.plot(timestamps, altA, color="#000000", label="Raw Altitude (altA)")
plt.plot(timestamps, filtA, color="#FF0000", label="Cube Regression (filtA)")
plt.gca().xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))
plt.legend()
plt.grid()
plt.xlabel("Time")
plt.ylabel("Altitude (meter)")
plt.show()


plt.figure("altB", figsize=(20,8))
plt.title(" Sensor-B (" + start_time.date().strftime("%Y-%m-%d") + ")")
plt.plot(timestamps, altB, color="#000000", label="Raw Altitude (altB)")
plt.plot(timestamps, filtB, color="#FF0000", label="Moving Average, W=4 (filtB)")
plt.gca().xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))
plt.legend()
plt.grid()
plt.xlabel("Time")
plt.ylabel("Altitude (meter)")
plt.show()


plt.figure("diff", figsize=(20,8))
plt.title(" Altitude Differnce (A-B) (" + start_time.date().strftime("%Y-%m-%d") + ")")
plt.plot(diffAB, color="#000000", label="Raw Altitude (altB)")
plt.gca().xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))
plt.legend()
plt.grid()
plt.xlabel("Time")
plt.ylabel("Altitude (meter)")
plt.show()

