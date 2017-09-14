from __future__ import print_function

# We'll need numpy for some mathematical operations
import numpy as np
import scipy.io
import pickle
import os
import librosa
# And the display module for visualization
import librosa.display

# mel-spectrogram parameters
# SR = 12000
# N_FFT = 512
# HOP_LEN = 256
# DURA = 29.12

def feature_extraction(indir, audio_name, ext, outdir, SR, N_FFT, HOP_LEN, DURA):

    # Load audio
    src, sr = librosa.load(indir+'/'+audio_name+ext, sr=SR)


    # Trim audio
    n_sample = src.shape[0]
    n_sample_wanted = int(DURA * SR)
    if n_sample < n_sample_wanted:  # if too short
        src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
    elif n_sample > n_sample_wanted:  # if too long
        src = src[(n_sample - n_sample_wanted) / 2:(n_sample + n_sample_wanted) / 2]


    # Perform harmonic percussive source separation (HSS)
    y_harmonic, y_percussive = librosa.effects.hpss(src)
    logam = librosa.logamplitude

    # for Spectral features
    for i in range(2):
        if i == 0:
            y = y_harmonic
        else:
            y = y_percussive

        fv = logam(librosa.feature.chroma_stft(y=y, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT), ref_power=np.max)
        if i == 0:
            fv_total = fv
        else:
            fv_total = np.vstack((fv_total, fv))

        fv = logam(librosa.feature.chroma_cens(y=y, sr=SR, hop_length=HOP_LEN), ref_power=np.max)
        fv_total = np.vstack((fv_total, fv))

        fv = logam(librosa.feature.melspectrogram(y=y, hop_length=HOP_LEN, n_fft=N_FFT, n_mels=96), ref_power=np.max)
        fv_total = np.vstack((fv_total, fv))

        fv_mfcc = librosa.feature.mfcc(y=y, sr=SR, hop_length=HOP_LEN)
        fv = logam(fv_mfcc, ref_power=np.max)
        fv_total = np.vstack((fv_total, fv))
        fv = logam(librosa.feature.delta(fv_mfcc), ref_power=np.max)
        fv_total = np.vstack((fv_total, fv))
        fv = logam(librosa.feature.delta(fv_mfcc, order=2), ref_power=np.max)
        fv_total = np.vstack((fv_total, fv))

        fv = logam(librosa.feature.rmse(y=y, hop_length=HOP_LEN, n_fft=N_FFT), ref_power=np.max)
        fv_total = np.vstack((fv_total, fv))

        fv = logam(librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT), ref_power=np.max)
        fv_total = np.vstack((fv_total, fv))

        fv = logam(librosa.feature.spectral_bandwidth(y=y, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT), ref_power=np.max)
        fv_total = np.vstack((fv_total, fv))

        fv = logam(librosa.feature.spectral_rolloff(y=y, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT), ref_power=np.max)
        fv_total = np.vstack((fv_total, fv))

        fv = logam(librosa.feature.poly_features(y=y, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT), ref_power=np.max)
        fv_total = np.vstack((fv_total, fv))

        fv = logam(librosa.feature.poly_features(y=y, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT, order=2), ref_power=np.max)
        fv_total = np.vstack((fv_total, fv))

        fv = logam(librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LEN, frame_length=N_FFT), ref_power=np.max)
        fv_total = np.vstack((fv_total, fv))

    # Feature aggregation
    fv_mean = np.mean(fv_total, axis=1)
    fv_var = np.var(fv_total, axis=1)
    fv_amax = np.amax(fv_total, axis=1)
    fv_aggregated = np.hstack((fv_mean, fv_var))
    fv_aggregated = np.hstack((fv_aggregated, fv_amax))

    # # for tempo features
    # tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=SR)
    # fv_aggregated = np.hstack((fv_aggregated, tempo))

    # # for Rhythm features
    # oenv = librosa.onset.onset_strength(y=y_percussive, sr=SR)
    # tempo = librosa.feature.tempogram(onset_envelope=oenv, sr=SR)
    # tempo_mean = np.mean(tempo, axis=1)
    # tempo_var = np.var(tempo, axis=1)
    # tempo_amax = np.amax(tempo, axis=1)
    # tempo_aggregated = np.hstack((tempo_mean, tempo_var))
    # tempo_aggregated = np.hstack((tempo_aggregated, tempo_amax))

    # for pickle
    pklName = outdir + "/" +  audio_name + '.pkl'
    f = open(pklName, 'wb')
    pickle.dump(fv_aggregated, f)
    f.close()

   
    print(audio_name)
