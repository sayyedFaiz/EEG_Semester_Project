import sys
import os
sys.path.insert(0,'../exercises')
import mne, osfclient, mne_bids
import ccs_eeg_utils
import numpy as np
import pandas as pa
from matplotlib import pyplot as plt
from mne_bids import (BIDSPath, read_raw_bids)

class Preprocess:
    def __init__(self, subjectId) -> None:
        self.subjectId = subjectId

    def loadData(self):
        #Path of the datasets
        bids_root = "./local/bids"
        bids_path = BIDSPath(subject=self.subjectId,task="P3",session="P3",
                            datatype='eeg', suffix='eeg',
                            root=bids_root)
        self.raw = read_raw_bids(bids_path)
        ccs_eeg_utils.read_annotations_core(bids_path,self.raw)

        self.raw.load_data() #Read the file

    def reReference(self):
        mne.set_eeg_reference(inst=self.raw, ref_channels=["P9","P10"])


    def filter_raw(self):
        self.raw.plot_psd(area_mode='range', tmax=10.0, average=False,xscale="log")
        self.raw.filter(0.5, 50, fir_design='firwin')
        self.raw.plot_psd(area_mode='range', tmax=10.0, average=False,xscale="log")




    def createEpochs(self):
        evts,evts_dict = mne.events_from_annotations(self.raw)
        wanted_keys = [e for e in evts_dict.keys() if "stimulus" in e]
        evts_dict_stim=dict((k, evts_dict[k]) for k in wanted_keys if k in evts_dict)
        epochs = mne.Epochs(self.raw,evts,evts_dict_stim,tmin=-0.2,tmax=0.8, baseline=(-0.2,0))
        epochs.average().plot()
        return epochs

    def compareEpochs(self):
        evts,evts_dict = mne.events_from_annotations(self.raw)
        wanted_keys = [e for e in evts_dict.keys() if "stimulus" in e]
        evts_dict_stim=dict((k, evts_dict[k]) for k in wanted_keys if k in evts_dict)
        epochs_raw = mne.Epochs(self.raw,evts,evts_dict_stim,tmin=-0.2,tmax=0.8,reject_by_annotation=False, baseline=(-0.2,0))
        epochs = mne.Epochs(self.raw,evts,evts_dict_stim,tmin=-0.2,tmax=0.8,reject_by_annotation=True, baseline=(-0.2,0))
        mne.viz.plot_compare_evokeds({'raw':epochs_raw.average(),'clean':epochs.average()})


    def saveBadAnnotations(self):
        bad_ix = [i for i, a in enumerate(self.raw.annotations) if a['description'] == "BAD_"]
        bad_annotations = self.raw.annotations[bad_ix]
        csv_filename = f"sub-{self.subjectId}_task-P3_badannotations.csv"
        bad_annotations.save(csv_filename , overwrite=True)
        annotations = mne.read_annotations(csv_filename)
        self.raw.annotations.append(annotations.onset,annotations.duration,annotations.description)




    def computeICA(self):
        self.raw.set_channel_types({'HEOG_left': 'eog', 'HEOG_right': 'eog', 'VEOG_lower': 'eog'})
        self.raw.set_montage('standard_1020', match_case=False)
        ica = mne.preprocessing.ICA(method="infomax")
        ica.fit(self.raw,verbose=True)
        return ica

    def getEvokedResponses(self , epochs):
        target = epochs[["stimulus:{}{}".format(k,k) for k in [1,2,3,4,5]]].average()
        distractor = epochs[["stimulus:{}{}".format(k,j) for k in [1,2,3,4,5] for j in [1,2,3,4,5] if k!=j]].average()
        evokeds = dict(target=target, distractor=distractor)

        mne.viz.plot_compare_evokeds(evokeds, combine='mean',picks=['Cz','Pz'])

        return target , distractor

    def getPeakValues(self, target):
        # Get peak amplitude and latency from the selected time window for target with selected channel 'Pz'
        target_Pz = target.copy().pick("Pz")
        good_tmin, good_tmax = 0.300, 0.600 #Recommended Measurement Windows (ms) 300ms-600 ms
        ch, lat, amp = target_Pz.get_peak(ch_type='eeg', tmin=good_tmin, tmax=good_tmax, mode='pos', return_amplitude=True)
        print("** PEAK MEASURES FOR ONE CHANNEL FROM A GOOD TIME WINDOW **")
        print(f'Subject: {"Subject"+self.subjectId}')
        print(f'Channel: {ch}')
        print(f'Time Window: {good_tmin * 1e3:.3f} - {good_tmax * 1e3:.3f} ms')
        print(f'Peak Latency: {lat * 1e3:.3f} ms')
        print(f'Peak Amplitude: {amp * 1e6:.3f} µV')
        # Get BAD peak measures
        bad_tmin, bad_tmax = 0.600, 0.799
        ch, bad_lat, bad_amp = target_Pz.get_peak(ch_type='eeg', tmin=bad_tmin, tmax=bad_tmax, mode='pos', return_amplitude=True)
        print("** PEAK MEASURES FOR ONE CHANNEL FROM A BAD TIME WINDOW **")
        print(f'Subject: {"Subject"+self.subjectId}')
        print(f'Channel: {ch}')
        print(f'Time Window: {bad_tmin * 1e3:.3f} - {bad_tmax * 1e3:.3f} ms')
        print(f'Peak Latency: {bad_lat * 1e3:.3f} ms')
        print(f'Peak Amplitude: {bad_amp * 1e6:.3f} µV')

        fig, axs = plt.subplots(nrows=2, ncols=1, layout='tight')
        words = (('Bad', 'missing'), ('Good', 'finding'))
        times = (np.array([bad_tmin, bad_tmax]), np.array([good_tmin, good_tmax]))
        colors = ('C1', 'C0')
        for ix, ax in enumerate(axs):
            title = '{} time window {} peak'.format(*words[ix])
            target_Pz.plot(axes=ax, time_unit='ms', show=False, titles=title)
            ax.plot(lat * 1e3, amp * 1e6, marker='*', color='C6')
            ax.axvspan(*(times[ix] * 1e3), facecolor=colors[ix], alpha=0.3)


    def getPeaks_P3(self,target, distractor):

        Channels = ['Cz','Pz']
        target_list = []
        distractor_list = []

        for ch in Channels:
            good_tmin, good_tmax = 0.300, 0.600
            ch_target, lat_target,amp_target = target.copy().pick(ch).get_peak(ch_type='eeg', tmin=good_tmin, tmax=good_tmax, mode='pos', return_amplitude=True)
            target_list.append(amp_target)

            print("Target:")
            print("** PEAK MEASURES FOR [''Cz','Pz''] FROM A GOOD TIME WINDOW **")
            print(f'Channel: {ch_target}')
            print(f'Time Window: {good_tmin * 1e3:.3f} - {good_tmax * 1e3:.3f} ms')
            print(f'Peak Latency: {lat_target * 1e3:.3f} ms')
            print(f'Peak Amplitude: {amp_target * 1e6:.3f} µV')

            ch_distractor, lat_distractor,amp_distractor = distractor.copy().pick(ch).get_peak(ch_type='eeg', tmin=good_tmin, tmax=good_tmax, mode='pos', return_amplitude=True)
            distractor_list.append(amp_distractor)

            print("Distractor:")
            print("** PEAK MEASURES FOR [''Cz','Pz''] FROM A GOOD TIME WINDOW **")
            print(f'Channel: {ch_distractor}')
            print(f'Time Window: {good_tmin * 1e3:.3f} - {good_tmax * 1e3:.3f} ms')
            print(f'Peak Latency: {lat_distractor * 1e3:.3f} ms')
            print(f'Peak Amplitude: {amp_distractor * 1e6:.3f} µV')

        return target_list , distractor_list


