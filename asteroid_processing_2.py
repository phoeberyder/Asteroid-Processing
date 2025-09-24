import pandas as pd
from unpack_vdif import readheader, readframes, sortframes, unpacksamps
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, fft, fftshift
from datetime import datetime
from datetime import timedelta
from scipy import constants


def resample_deldots(filename):
    '''# This gives you a pandas 'series' jb_deldot with the data you need indexed by time, so that you can now say 
# deldot = jb_deldot[row_time]'''
    df=pd.read_csv(filename,header=None, delimiter=',') # read in the file to a dataframe df

    df['datetime']=pd.to_datetime(df[0])  #convert the first col to a proper datetime in a new column
    df=df.set_index('datetime')  # make this the index
    series=df[2] # make series from the deldot column, change the column number depending on whats in your file
    
    deldot=series.resample("1s").interpolate('linear')  #re-sample to 1 sec intervals
    return df, deldot

df1, jb_deldot = resample_deldots('/Users/user/Documents/VS Code/Radar/Asteroid-Processing/jodrell-2006wb.txt')
df2, rob_deldot = resample_deldots('/Users/user/Documents/VS Code/Radar/Asteroid-Processing/robledo-2006wb.txt')

infilename = '/Users/user/Documents/VS Code/Radar/Asteroid-Processing/DD18004_20241126_lo1_7167MHz_2026WB_14.vdif'
infile = open(infilename)
header = readheader(infile)
framedata, seconds, framenums, threads = readframes(infile, header)
infile.close() # finished with file
threaddata = sortframes(framedata, seconds, framenums, threads)

pola = unpacksamps(threaddata[0,:], header['nbits'], header['dtype'])

fft_length = 1

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
# I know this works
# for i in range(height):
#     row_time=start_time+timedelta(seconds=row_length*i)
#     deldot = rob_deldot[row_time] + jb_deldot[row_time]
#     Delta_f=f_0*((deldot*1e3)/c)
#     shift.append(int(Delta_f))

#     temp = rfft(dat[i])
#     shifted=np.roll(temp, int(Delta_f))
#     result[i] = shifted
# T = 60
# fs = 4e6
# t = np.linspace(0, T, int(fs * T), endpoint=False)
BW = 2e6
for i in range(height):
    row_time=start_time+timedelta(seconds=row_length*i)
    deldot = rob_deldot[row_time] + jb_deldot[row_time]
    Delta_f=f_0*((deldot*1e3)/c)
    shift.append(int(Delta_f))
    # a = Delta_f/row_time
    # negative_chirp = np.exp(-1j*2*np.pi* (a* t**2))
    # deramp = shift*negative_chirp
    cdat[i] = dat[i]
    temp = fftshift(fft(cdat[i]))
    chan_shift = int((Delta_f/BW)*points/2)
    shifted=np.roll(temp, int(Delta_f))
    result[i] = shifted



# print(len(shift))

print('shifted')
# shifted=np.roll(result,shift, axis=1)
# shifted_r = shifted.real
centre = (len(shifted)/4)*3
low = int(centre-200)
high = int(centre+200)


# p = shifted_r[low:high]
# plt.plot(p)

# plt.imshow(shifted, origin='lower', cmap='rainbow')
# plt.show()



# shifted = (np.abs(result))**2


# spec = shifted.sum(axis=0)
# spec = spec[1:]
# # sd = np.std(spec) 
# # spec /= sd
# # print(sd)
# centre = len(spec)/4
# low = int(centre-200)
# high = int(centre+200)

# #uncomment for 1d spec plot
# p = spec[low:high]

# print(spec)
# print('about to plot')
# plt.plot(p)

# plt.xlabel("samples")
# plt.ylabel("power")
# plt.show()

# plt.imshow(shifted[:, low:high], origin='lower', cmap='plasma')
# plt.colorbar()

result_sub=result[:,low:high]
power_sub=(np.abs(result_sub))**2
spec_sub=power_sub.sum(axis=0)


plt.plot(spec_sub)
plt.show()

plt.savefig('spectrum.png')