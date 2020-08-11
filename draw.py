import numpy as np
import minisom as som
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import json

# Load UniverseMachine
num = 0
filename = "../full-photometry-catalogs/survey_PFS_z0.69-1.71_x280.00_y280.00_%d"%num
# Load the zeroth realization data and meta data
cat=np.load(filename+".npy")
cat_meta=json.load(open(filename+".json"))
# NEW: Load spectral features for the zeroth realization only
cat_specprop=np.load(filename+".specprop")

bands = ['m_u','m_b','m_v','m_g','m_r','m_i','m_z','m_y','m_j']
data = np.array([(cat[iband]-np.mean(cat[iband]))/np.std(cat[iband]) for iband in bands]).T
redshift = cat['redshift']
#print(cat_specprop.dtype.names)
linename = 'f_OIII4960'
target = cat_specprop[linename]


bins = np.linspace(0,4e-15,500)
plt.hist(target,bins=bins,density=True,label='Raw')
print(np.average(target))
#plt.yscale('log')
#plt.savefig(linename+'-test.png')
#plt.clf()

size = 40
train = True

som = som.MiniSom(size,size,len(data[0]),sigma=1,learning_rate=0.1)
som._weights = np.load('som-%d.npy'%size)
utils_weights = np.load('utils-%d.npy'%size)
weights = utils_weights.flatten()


distribution = [[]]
for i in range(size**2-1):
    distribution = distribution+[[]] 
for cnt,xx in enumerate(data):
    w = som.winner(xx) 
    distribution[size*w[0]+w[1]].append(cnt)

chosen_bins = np.random.choice(range(size**2),100000,p=weights/np.sum(weights)) #weighted
#chosen_bins = np.random.choice(range(size**2),20000) #uniformly

chosen = []
for i in range(size**2):
    chosen.append(np.random.choice(distribution[i],np.count_nonzero(chosen_bins==i)))

chosen_utils = []
for ibin in chosen:
    for iarg in ibin:
        chosen_utils.append(target[iarg])
print(max(chosen_utils),min(chosen_utils))
plt.hist(chosen_utils,bins=bins,density=True,label='Selected',alpha=0.5)
print(np.average(chosen_utils))
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.title('Density')
plt.savefig('sampling.png')
print('Finish')



