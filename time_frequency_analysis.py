import numpy as np
import scipy
import mne
import sys
sys.path.insert(0,"../exercises/")
from matplotlib import pyplot as plt
import ccs_eeg_utils

class Tf_analysis:
    def __init__(self, subjectId) -> None:
        self.subjectId = subjectId

    def time_frequency_analysis(self, bids_root):
        epochs = ccs_eeg_utils.get_TF_dataset(  subject_id = self.subjectId,bids_root = bids_root)
        freqs = np.logspace(*np.log10([5, 80]), num=25)
        n_cycles = freqs/2
        power_total = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False,n_jobs=4,average=True, picks='Cz')
        print(power_total)
        power_total.plot_topo(baseline=[-.5,0],mode="logratio",vmin=-2,vmax=2,title="Power logratio Topography")

        power_total.plot(baseline=None,vmin=-2*10e-9,vmax=2*10e-9,picks='Cz',title="Power Topography without basline")
        power_total.plot(baseline=[-.5,0],mode='percent',vmin=-4,vmax=4,picks='Cz',title="Power Topography with basline")

        epochs_induced = epochs.copy()
        #epochs_induced._data = epochs_induced._data  - epochs_induced.average().data # but we are using the offocial way here
        epochs_induced.subtract_evoked()
        power_induced = mne.time_frequency.tfr_morlet(epochs_induced, freqs=freqs, n_cycles=n_cycles, return_itc=False,n_jobs=1,average=True,picks="Cz")

        power_evoked = mne.combine_evoked([power_total,power_induced],weights=[1,-1])
        mode = "percent"
        bsl = [-0.5,0]
        cmin = -3
        cmax = -cmin
        power_total.plot(baseline=bsl,mode=mode,picks='Cz',vmin=cmin,vmax=cmax,title="Power_total Topography ")
        power_induced.plot(baseline=bsl,mode=mode,picks='Cz',vmin=cmin,vmax=cmax,title="Power_induced Topography ")
        power_evoked.plot(baseline=bsl,mode=mode,picks='Cz',vmin=cmin,vmax=cmax,title="Power_invoked Topography ")

        target = epochs["response:201"].average()
        distractor = epochs["response:202"].average()
        target.plot_topomap(times = (0.01, 0.12, 0.17, 0.3), ch_type='eeg', show_names=True, colorbar=False,size=2, res=128,time_unit='ms', image_interp='linear')
        distractor.plot_topomap(times = (0.01, 0.12, 0.17, 0.3), ch_type='eeg', show_names=True, colorbar=False,size=2, res=128,time_unit='ms', image_interp='linear')


