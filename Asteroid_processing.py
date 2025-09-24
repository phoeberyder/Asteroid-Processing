#importing required packages

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, fft

# infilename = '/Users/user/Documents/VS Code/Radar/TSSat_20240321_1440_Cm_1333_astra1n_on.vdif'     #data set on which Justin's code works - useful for reference of what I am aiming for
infilename = '/Users/user/Documents/VS Code/Radar/DD18004_20241126_da_7167MHz_2026WB_10.vdif'

# Read header from an open VDIF file, and return a dict containing header data.
def readheader(infile):
  # Necessary munging of binary header to put it in the same layout as figure 3 in the format spec.
  # This was a bugger to get right.
  words = np.fromfile(infile, dtype=np.uint8, count=4*4)
  words = np.reshape(words, [4,4])[:,::-1]
  words = np.unpackbits(words, axis=1)

  header = {}

  # Parse word 0.
  header['invalid'] = words[0][0] # invalidity flag
  header['legacy'] = words[0][1] # legacy format flag
  words[0][:2] = 0 # erase flags to avoid contaminating first byte of second counter
  vals = np.packbits(words[0])
  header['seconds'] = sum(256**np.arange(3,-1,-1) * vals)

  # Parse word 1.
  assert words[1][0] == words[1][1] == 0, 'Unassigned bits should be zero.'
  header['epoch'] = np.packbits(words[1][:8])[0] # epoch in 6-month intervals from 1 Jan 2000
  vals = np.packbits(words[1][8:]) # remainder of word breaks cleanly at end of first byte
  header['framenum'] = sum(256**np.arange(2,-1,-1) * vals) # frame # within second

  # Parse word 2.
  vals = words[2][:3]
  header['version'] = sum(2**np.arange(2,-1,-1) * vals)
  assert header['version'] == 0, 'VDIF header version not supported.'
  vals = words[2][3:8] # bits in header actually represent log2(nchan)
  header['nchan'] = 2**sum(2**np.arange(4,-1,-1) * vals) # number of channels
  vals = np.packbits(words[2][8:]) # word breaks cleanly at end of first byte again
  header['framelen'] = sum(256**np.arange(2,-1,-1) * vals)*8 # frame length in bytes

  # Parse word 3.
  header['dtype'] = 'C' if words[3][0] else 'R' # complex or real data
  vals = words[3][1:6]
  header['nbits'] = sum(2**np.arange(4,-1,-1) * vals)+1 # sample size in bits
  vals = words[3][6:16]
  header['thread'] = sum(2**np.arange(9,-1,-1) * vals) # thread ID
  vals = np.packbits(words[3][16:]) # word breaks cleanly at end of second byte this time
  header['station'] = sum(256**np.arange(1,-1,-1) * vals) # station ID

  # Parse remaining words if necessary.
  if not header['legacy']:
    xwords = np.fromfile(infile, dtype=np.uint8, count=4*4)
    assert not xwords.sum(), 'Not set up to process extended user data.'

  # Derived values.
  header['headerlen'] = 16 + 16*(1 - header['legacy']) # bytes; full length of header
  header['datalen'] = header['framelen'] - header['headerlen'] # bytes; length of data only
  # Nicely-formatted dates, etc., would go here, derived from 'epoch' and 'seconds'.

  return header

# Read frames from an open VDIF file, up to the entire file, from specified starting frame.
def readframes(infile, header, nframes=99999999999, initframe=0):
  # Find file size.
  infile.seek(0,2)
  filesize = infile.tell()

  assert not filesize % header['framelen'], 'File size must be evenly divisible by frame length.'

  # Reading will start from initial frame.
  infile.seek(initframe*header['framelen'],0)
  # Do not read beyond end of file.
  nframes = min(nframes, filesize // header['framelen'] - initframe)

  raw = np.fromfile(infile, np.uint8, count=nframes*header['framelen'])
  raw = np.reshape(raw, [nframes, header['framelen']])

  # Extract parameters that vary between frames.
  # First, seconds from reference epoch.
  seconds = raw[:,3::-1]
  seconds = np.unpackbits(seconds, axis=1)
  seconds[:,:2] = 0 # blank out flags
  seconds = np.packbits(seconds, axis=1).astype(np.int64)
  seconds = 256**3*seconds[:,0] + 256**2*seconds[:,1] + 256**1*seconds[:,2] + 256**0*seconds[:,3]
  # Second, frame number within second.
  framenums = raw[:,6:3:-1].astype(np.int64)
  framenums = 256**2*framenums[:,0] + 256**1*framenums[:,1] + 256**0*framenums[:,2]
  # Third, thread ID.
  threads = raw[:,15:13:-1]
  threads = np.unpackbits(threads, axis=1)
  threads[:,:6] = 0 # blank out flag and sample size
  threads = np.packbits(threads, axis=1).astype(np.int64)
  threads = 256**1*threads[:,0] + 256**0*threads[:,1]

  framedata = raw[:,header['headerlen']:]

  return framedata, seconds, framenums, threads



# Sort frame data into separate sequences for each thread.
def sortframes(framedata, seconds, framenums, threads):
  ithreads = np.unique(threads)

  # Find separate index list for frames for each thread.
  lexinds = np.lexsort(( framenums, seconds ))
  inds = []
  for ithread in ithreads:
    tinds = lexinds[np.where(threads[lexinds] == ithread)]
    inds.append(tinds)

  # Trim mismatched preceding frames.
  fseconds   = seconds  [[tinds[0] for tinds in inds]]
  fframenums = framenums[[tinds[0] for tinds in inds]]
  while len(np.unique(fseconds)) > 1 or len(np.unique(fframenums)) > 1:
    ftind = np.lexsort(( fframenums, fseconds ))[0]
    inds[ftind] = inds[ftind][1:]
    fseconds   = seconds  [[tinds[0] for tinds in inds]]
    fframenums = framenums[[tinds[0] for tinds in inds]]

  # Trim mismatched following frames.
  fseconds   = seconds  [[tinds[-1] for tinds in inds]]
  fframenums = framenums[[tinds[-1] for tinds in inds]]
  while len(np.unique(fseconds)) > 1 or len(np.unique(fframenums)) > 1:
    ftind = np.lexsort(( fframenums, fseconds ))[-1]
    inds[ftind] = inds[ftind][:-1]
    fseconds   = seconds  [[tinds[-1] for tinds in inds]]
    fframenums = framenums[[tinds[-1] for tinds in inds]]

  for ithread,tinds in zip(ithreads,inds):
    assert max(np.diff(seconds[tinds])) <= 1, 'Missing data in thread %d.' % ithread
  # Note that this check will not notice if the missing frame is the final frame in a second.

  assert len(np.unique([len(tinds) for tinds in inds])) == 1, 'Different numbers of frames in different threads.'
  inds = np.array(inds) # we can do this now that we know the array is perfectly rectangular

  nthreads = len(ithreads)
  assert ithreads.max() == nthreads-1, 'Discontiguous thread numbering.'
  # Assumed from this point onward.

  # Copy out thread data.
  framelen = np.shape(framedata)[1]
  nframes = np.shape(inds)[1]
  threaddata = np.zeros([nthreads,nframes,framelen], dtype=framedata.dtype)

  for ithread,tinds in enumerate(inds):
    for iframe,tind in enumerate(tinds):
      threaddata[ithread,iframe,:] = framedata[tind,:]

  threaddata = np.reshape(threaddata, [len(ithreads), nframes*framelen])

  return threaddata

def unpacksamps(samps, nbits, dtype):
  assert nbits == 2, 'Unpacking procedure is hard-coded for 2-bit samples.'
  assert dtype == 'R', 'Unpacking procedure assumes real samples.'
  assert len(np.shape(samps)) == 1, 'Unpacking procedure is written for one-dimensional arrays.'

  nmax = 2**24 # max bytes to unpack at a time

  if len(samps) <= nmax:
    samps = np.unpackbits(samps)
    samps = 2*samps[0::2] + samps[1::2]

    # Reorder 1-byte chunks.
    samps = np.reshape(samps, [int(len(samps)/4), 4])
    samps = samps[:,::-1]
    samps = np.reshape(samps, np.size(samps))

    return samps
  
   # We have a large amount of data.  Do it a chunk at a time.

  oldsamps = samps
  samps = np.zeros(len(oldsamps)*4, dtype=np.uint8)
  isamp = 0

  while len(oldsamps):
    tmpsamps = oldsamps[:nmax]
    oldsamps = oldsamps[nmax:]

    tmpsamps = unpacksamps(tmpsamps, nbits, dtype)

    samps[isamp:isamp+len(tmpsamps)] = tmpsamps
    isamp += len(tmpsamps)

  return samps
# Open file and get header info.
infile = open(infilename)
header = readheader(infile)

# Get data from file.
framedata, seconds, framenums, threads = readframes(infile, header)
infile.close() # finished with file

# Sort into streams for separate threads.
threaddata = sortframes(framedata, seconds, framenums, threads)

# Unpack 2-bit data for first thread(/polarisation).
pola = unpacksamps(threaddata[0,:], header['nbits'], header['dtype'])
# At this point, we have true, sequential sample values for a single polarisation.
#polb = unpacksamps(threaddata[1,:], header['nbits'], header['dtype'])


#first chopping 

pola_reduced = pola[0:1068000000]
result = np.zeros((267, 2000001))
dat = pola_reduced.reshape((267, 4000000))


for i in range(267):
    result[i] = rfft(dat[i])

out = (np.abs(result))**2
# plt.imshow(out[:, ::1000], vmin= 0.8e6, vmax=10000000, cmap='rainbow', origin='lower')
# plt.colorbar()

midpoint = 2000001/2
#using speed at 18:15 
s = 0.2327428 *10**3 * 2 / 3e8 *  7167e6
s2 = 0.2308106 *10**3 *2 /3e8 *7167e6
d1 = midpoint - (s)
d2 = midpoint + (s)

#finding plotting boundaries for both sides of the centre

low = int(d1-500)
high = int(d1+500)

low2 = int(d2-500)
high2 = int(d2+500)

plt.imshow(out[:, low:high], vmin= 1e5, vmax=2e7, cmap='rainbow', origin='lower')
plt.colorbar()
plt.show()

plt.imshow(out[:, low2:high2], vmin= 0.9e6, vmax=10e6, cmap='plasma', origin='lower')
plt.colorbar()
plt.show()

# deramping

fs = 4e6 
T = 60
f0 = 7166988879.549016 
f1 = 7166988971.869532 
t = np.linspace(0, T, int(fs * T), endpoint=False)
a = (f1-f0)/T
negative_chirp = np.exp(-1j*2*np.pi* (f0*t + 0.5*a* t**2))

minute = pola[0:240000000]
deramp = minute*negative_chirp

deramp = deramp.reshape((60, 4000000))
deramp_out = np.zeros((60, 4000000))
for i in range(60):
    deramp_out[i] = fft(deramp[i])
deramp_out = (np.abs(deramp_out))**2

plt.imshow(deramp_out[:, ::10000], vmin= 1e2, vmax=5e6, cmap='rainbow', origin='lower')
plt.colorbar()