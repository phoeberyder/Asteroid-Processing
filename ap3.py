import os
import pandas as pd
from unpack_vdif import readheader, readframes, sortframes, unpacksamps
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from datetime import datetime
from datetime import timedelta
from scipy import constants

# fnames = []

# for filename in os.listdir('/share/nas2/pryder/2006WB/vdifs'):
#     f = os.path.join('/share/nas2/pryder/2006WB/vdifs', filename)
#     fnames.append(f)

fnames = ['/Users/user/Downloads/DD18004_20241126_lo2_7167MHz_2026WB_13.vdif']

def resample_deldots(filename):
    '''# This gives you a pandas 'series' jb_deldot with the data you need indexed by time, so that you can now say 
# deldot = jb_deldot[row_time]'''
    df=pd.read_csv(filename,header=None, delimiter=',') # read in the file to a dataframe df

    df['datetime']=pd.to_datetime(df[0])  #convert the first col to a proper datetime in a new column
    df=df.set_index('datetime')  # make this the index
    series=df[2]
    deldot=series.resample("1s").interpolate('linear')  #re-sample to 1 sec intervals
    return df, deldot

df1, jb_deldot = resample_deldots('./jodrell-2006wb-pr.txt')
df2, rob_deldot = resample_deldots('./robledo-2006wb-pr.txt')

infile = open(fnames[0])
header = readheader(infile)
framedata, seconds, framenums, threads = readframes(infile, header)
infile.close() # finished with file
threaddata = sortframes(framedata, seconds, framenums, threads)

pola = unpacksamps(threaddata[0,:], header['nbits'], header['dtype'])

fft_length = 16

points = 4000000*fft_length

height = len(pola)//points

reduced_length = height*points

zeros_length = int((points/2)+1)

pola_reduced = pola[0:reduced_length]
dat = pola_reduced.reshape((height, points))
cdat = np.zeros_like(dat, dtype=complex)

seconds=header['seconds']
yrs=header['epoch']/2

refdate=datetime(2017,1,1)
start_time=refdate+timedelta(seconds=float(seconds))
print(start_time.isoformat())

print('about to Fourier transform')
row_length = fft_length # 1 second
shift = []
c=constants.c
f_0 = 7166988879.549016
result = np.zeros((height, points), dtype=complex)

BW = 2e6
print("doing rows:",height)
for i in range(height):
    print("row:",i)
    txoffset=6.0
    row_time=start_time+timedelta(seconds=row_length*i)
    previous_row_time = start_time+timedelta(seconds=row_length*(i-1))
    next_row_time = start_time+timedelta(seconds=row_length*(i+1))
    deldot = rob_deldot[row_time+timedelta(seconds=-txoffset)] + jb_deldot[row_time]
    previous_deldot = rob_deldot[previous_row_time+timedelta(seconds=-txoffset)]+jb_deldot[previous_row_time]
    next_deldot = rob_deldot[next_row_time+timedelta(seconds=-txoffset)]+jb_deldot[next_row_time]
    next_delta_f = f_0*((next_deldot*1e3)/c)
    previous_delta_f = f_0*((previous_deldot*1e3)/c)
    a = (next_delta_f-previous_delta_f)/(2*row_length)
    Delta_f=f_0*(((deldot)*1e3)/c)
    t = np.linspace(0, fft_length, int(points))
    negative_chirp = np.exp(1j*2*np.pi* ((Delta_f*t)+(0.5*a* t**2)))
    cdat[i] = dat[i]
    cdat[i] = cdat[i]*negative_chirp
    temp_chirp = fftshift(fft(cdat[i]))
    result[i] = temp_chirp

import plotly.express as px
mid=int(result.shape[1]*0.75)
print(mid)
span=1000
result_sub=result[:,mid-span:mid+span]
result_sub=(np.abs(result_sub))**2
spec=result_sub.sum(axis=0)
fig=px.line(spec)
fig.show()



chan_offset=np.argmax(spec)-span
hz_offset=2*BW*(chan_offset/points)
print("offset in channels:",chan_offset)
print("offset in Hz:      ",hz_offset)
