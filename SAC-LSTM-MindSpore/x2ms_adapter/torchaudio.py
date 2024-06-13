#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
import mindspore
import mindspore.dataset.audio as audio
import numpy as np
from mindspore.dataset.audio import BorderType, DensityFunction, Modulation, \
    Interpolation, ResampleMethod, NormType, MelType, NormMode, WindowType, GainType, FadeShape
from .torch_base_api import hann_window, bartlett_window, blackman_window, hamming_window


def allpass_biquad(waveform, sample_rate, central_freq, Q=0.707):
    waveform = get_numpy_data(waveform)
    return mindspore.Tensor(audio.AllpassBiquad(sample_rate, central_freq, Q)(waveform))


def band_biquad(waveform, sample_rate, central_freq, Q=0.707, noise=False):
    waveform = get_numpy_data(waveform)
    return mindspore.Tensor(audio.BandBiquad(sample_rate, central_freq, Q, noise)(waveform))


def bandpass_biquad(waveform, sample_rate, central_freq, Q=0.707, const_skirt_gain=False):
    waveform = get_numpy_data(waveform)
    return mindspore.Tensor(audio.BandpassBiquad(sample_rate, central_freq, Q, const_skirt_gain)(waveform))


def bandreject_biquad(waveform, sample_rate, central_freq, Q=0.707):
    waveform = get_numpy_data(waveform)
    return mindspore.Tensor(audio.BandrejectBiquad(sample_rate, central_freq, Q)(waveform))


def bass_biquad(waveform, sample_rate, gain, central_freq=100, Q=0.707):
    waveform = get_numpy_data(waveform)
    return mindspore.Tensor(audio.BassBiquad(sample_rate, gain, central_freq, Q)(waveform))


def biquad(waveform, b0, b1, b2, a0, a1, a2):
    waveform = get_numpy_data(waveform)
    return mindspore.Tensor(audio.Biquad(b0, b1, b2, a0, a1, a2)(waveform))


def deemph_biquad(waveform, sample_rate):
    waveform = get_numpy_data(waveform)
    return mindspore.Tensor(audio.DeemphBiquad(sample_rate)(waveform))


def equalizer_biquad(waveform, sample_rate, center_freq, gain, Q=0.707):
    waveform = get_numpy_data(waveform)
    return mindspore.Tensor(audio.EqualizerBiquad(sample_rate, center_freq, gain, Q)(waveform))


def highpass_biquad(waveform, sample_rate, cutoff_freq, Q=0.707):
    waveform = get_numpy_data(waveform)
    return mindspore.Tensor(audio.HighpassBiquad(sample_rate, cutoff_freq, Q)(waveform))


def compute_deltas(waveform, win_length=5, mode='replicate'):
    mode_map = {"constant": BorderType.CONSTANT,
                "replicate": BorderType.EDGE,
                "reflect": BorderType.REFLECT,
                "symmetric": BorderType.SYMMETRIC,
                }
    wave_form = get_numpy_data(waveform)
    if mode not in mode_map.keys():
        raise NotImplementedError(f'Unsupported mode type {mode}')
    ms_obj = mindspore.dataset.audio.ComputeDeltas(
        win_length=win_length, pad_mode=mode_map.get(mode, BorderType.EDGE))
    return mindspore.Tensor(ms_obj(wave_form))


def contrast(waveform, enhancement_amount=75.0):
    wave_form = get_numpy_data(waveform)
    ms_obj = mindspore.dataset.audio.Contrast(enhancement_amount=enhancement_amount)
    return mindspore.Tensor(ms_obj(wave_form))


def dcshift(waveform, shift, limiter_gain=None):
    wave_form = get_numpy_data(waveform)
    ms_obj = mindspore.dataset.audio.DCShift(shift, limiter_gain)
    return mindspore.Tensor(ms_obj(wave_form))


def detect_pitch_frequency(waveform, sample_rate, frame_time=0.01, win_length=30, freq_low=85, freq_high=3400):
    wave_form = get_numpy_data(waveform)
    ms_obj = mindspore.dataset.audio.DetectPitchFrequency(
        sample_rate, frame_time=frame_time, win_length=win_length, freq_low=freq_low, freq_high=freq_high)
    return mindspore.Tensor(ms_obj(wave_form))


def dither(waveform, density_function='TPDF', noise_shaping=False):
    mode_map = {"TPDF": DensityFunction.TPDF,
                "RPDF": DensityFunction.RPDF,
                "GPDF": DensityFunction.GPDF,
                }
    wave_form = get_numpy_data(waveform)
    ms_obj = mindspore.dataset.audio.Dither(
        density_function=mode_map.get(density_function, DensityFunction.TPDF), noise_shaping=noise_shaping)
    return mindspore.Tensor(ms_obj(wave_form))


def x2ms_gain(waveform, gain_db=1.0):
    wave_form = get_numpy_data(waveform)
    ms_obj = mindspore.dataset.audio.Gain(gain_db)
    return mindspore.Tensor(ms_obj(wave_form))


def flanger(waveform, sample_rate, delay=0.0, depth=2.0, regen=0.0, width=71.0, speed=0.5, phase=25.0,
            modulation='sinusoidal', interpolation='linear'):
    map_mod = {
        "sinusoidal": Modulation.SINUSOIDAL,
        "triangular": Modulation.TRIANGULAR
    }
    map_inter = {
        "linear": Interpolation.LINEAR,
        "quadratic": Interpolation.QUADRATIC
    }
    wave_form = get_numpy_data(waveform)
    ms_obj = mindspore.dataset.audio.Flanger(
        sample_rate, delay, depth, regen, width, speed, phase,
        modulation=map_mod.get(modulation, Modulation.SINUSOIDAL),
        interpolation=map_inter.get(interpolation, Interpolation.LINEAR))
    return mindspore.Tensor(ms_obj(wave_form))


def phaser(waveform, sample_rate, gain_in=0.4, gain_out=0.74,
           delay_ms=3.0, decay=0.4, mod_speed=0.5, sinusoidal=True, ):
    wave_form = get_numpy_data(waveform)
    ms_obj = mindspore.dataset.audio.Phaser(sample_rate, gain_in, gain_out, delay_ms, decay, mod_speed, sinusoidal)
    return mindspore.Tensor(ms_obj(wave_form))


def overdrive(waveform, gain=20, colour=20):
    wave_form = get_numpy_data(waveform)
    ms_obj = mindspore.dataset.audio.Overdrive(gain=gain, color=colour)
    return mindspore.Tensor(ms_obj(wave_form))


def melscale_fbanks(n_freqs, f_min, f_max, n_mels, sample_rate, norm=None, mel_scale='htk'):
    norm = NormType.NONE if norm is None else NormType.SLANEY
    mel_scale = MelType.HTK if mel_scale == "htk" else MelType.SLANEY

    ms_obj = mindspore.dataset.audio.melscale_fbanks(
        n_freqs, f_min, f_max, n_mels, sample_rate, norm=norm, mel_type=MelType.HTK)
    return mindspore.Tensor(ms_obj)


def create_dct(n_mfcc, n_mels, norm=None):
    if norm is None:
        norm = NormMode.NONE
    else:
        norm = NormMode.ORTHO
    ms_obj = mindspore.dataset.audio.create_dct(n_mfcc, n_mels, norm=norm)
    return mindspore.Tensor(ms_obj)


def get_numpy_data(waveform):
    if isinstance(waveform, mindspore.Tensor):
        waveform = waveform.asnumpy()
    elif isinstance(waveform, np.ndarray):
        waveform = waveform
    else:
        raise NotImplementedError(f'Unsupported data type {type(waveform)}')
    return waveform


def lowpass_biquad(waveform, sample_rate: int, cutoff_freq: float, Q: float = 0.707):
    waveform = get_numpy_data(waveform)
    return mindspore.Tensor(mindspore.dataset.audio.LowpassBiquad(sample_rate, cutoff_freq, Q)(waveform))


def lfilter(waveform, a_coeffs, b_coeffs, clamp: bool = True, batching: bool = True):
    if not batching:
        raise NotImplementedError(f"batching = False is not implemented.")
    waveform = get_numpy_data(waveform)
    a_coeffs = get_numpy_data(a_coeffs).tolist()
    b_coeffs = get_numpy_data(b_coeffs).tolist()
    return mindspore.Tensor(mindspore.dataset.audio.LFilter(a_coeffs, b_coeffs, clamp)(waveform))


class MuLawDecoding(mindspore.dataset.audio.MuLawDecoding):
    def __init__(self, quantization_channels=256):
        super(MuLawDecoding, self).__init__(quantization_channels)

    def __call__(self, x_mu):
        x_mu = get_numpy_data(x_mu)
        return mindspore.Tensor(super(MuLawDecoding, self).__call__(x_mu))


class MuLawEncoding(mindspore.dataset.audio.MuLawEncoding):
    def __init__(self, quantization_channels=256):
        super(MuLawEncoding, self).__init__(quantization_channels)

    def __call__(self, x_mu):
        x_mu = get_numpy_data(x_mu)
        return mindspore.Tensor(super(MuLawEncoding, self).__call__(x_mu))


class Resample(mindspore.dataset.audio.Resample):
    def __init__(self, orig_freq: int = 16000, new_freq: int = 16000, resampling_method: str = 'sinc_interpolation',
                 lowpass_filter_width: int = 6, rolloff: float = 0.99, beta=None, *, dtype=None):
        if resampling_method == 'sinc_interpolation':
            resampling_method = ResampleMethod.SINC_INTERPOLATION
        else:
            resampling_method = ResampleMethod.KAISER_WINDOW
        self.dtype = dtype if dtype is not None else mindspore.float32
        super(Resample, self).__init__(orig_freq, new_freq, resampling_method, lowpass_filter_width, rolloff, beta)

    def __call__(self, waveform):
        waveform = get_numpy_data(waveform)
        return mindspore.Tensor(super(Resample, self).__call__(waveform), self.dtype)


class SlidingWindowCmn(mindspore.dataset.audio.SlidingWindowCmn):
    def __init__(self, cmn_window: int = 600, min_cmn_window: int = 100, center: bool = False, norm_vars: bool = False):
        super(SlidingWindowCmn, self).__init__(cmn_window, min_cmn_window, center, norm_vars)

    def __call__(self, waveform):
        waveform = get_numpy_data(waveform)
        return mindspore.Tensor(super(SlidingWindowCmn, self).__call__(waveform))


class SpectralCentroid(mindspore.dataset.audio.SpectralCentroid):
    def __init__(self, sample_rate: int, n_fft: int = 400, win_length=None, hop_length=None, pad=0,
                 window_fn=hann_window, wkwargs=None):
        dict_window_fn = {
            hann_window: WindowType.HANN,
            bartlett_window: WindowType.BARTLETT,
            blackman_window: WindowType.BLACKMAN,
            hamming_window: WindowType.HAMMING,
        }
        if window_fn in dict_window_fn.keys():
            window = dict_window_fn[window_fn]
        else:
            raise NotImplementedError(f"unsupported window_fn: {window_fn}")
        if wkwargs is not None:
            raise NotImplementedError(f"'wkwargs' is not implemented: {wkwargs}")
        super(SpectralCentroid, self).__init__(sample_rate, n_fft, win_length, hop_length, pad, window)

    def __call__(self, waveform):
        waveform = get_numpy_data(waveform)
        return mindspore.Tensor(super(SpectralCentroid, self).__call__(waveform)).squeeze()


class Spectrogram(mindspore.dataset.audio.Spectrogram):
    def __init__(self, n_fft=400, win_length=None, hop_length=None, pad=0, window_fn=hann_window, power=2.0,
                 normalized=False, wkwargs=None, center=True, pad_mode='reflect', onesided=True, return_complex=None):
        dict_window_fn = {
            hann_window: WindowType.HANN,
            bartlett_window: WindowType.BARTLETT,
            blackman_window: WindowType.BLACKMAN,
            hamming_window: WindowType.HAMMING,
        }
        if window_fn in dict_window_fn.keys():
            window = dict_window_fn[window_fn]
        else:
            raise NotImplementedError(f"unsupported window_fn: {window_fn}")
        if wkwargs is not None:
            raise NotImplementedError(f"'wkwargs' is not implemented: {wkwargs}")
        if return_complex is not None:
            raise NotImplementedError(f"'return_complex' is not implemented: {return_complex}")
        dict_pad_mode = {
            'reflect': BorderType.REFLECT,
            'edge': BorderType.EDGE,
            'constant': BorderType.CONSTANT,
            'symmetric': BorderType.SYMMETRIC,
        }
        if pad_mode in dict_pad_mode.keys():
            pad_mode = dict_pad_mode[pad_mode]
        else:
            raise NotImplementedError(f"unsupported pad_mode: {pad_mode}")            
        super(Spectrogram, self).__init__(n_fft, win_length, hop_length, pad, window, power, normalized, center,
                                          pad_mode, onesided)

    def __call__(self, waveform):
        waveform = get_numpy_data(waveform)
        return mindspore.Tensor(super(Spectrogram, self).__call__(waveform))


def treble_biquad(waveform, sample_rate: int, gain: float, central_freq: float = 3000, Q: float = 0.707):
    waveform = get_numpy_data(waveform)
    return mindspore.Tensor(mindspore.dataset.audio.TrebleBiquad(sample_rate, gain, central_freq, Q)(waveform))


class Vol(mindspore.dataset.audio.Vol):
    def __init__(self, gain: float, gain_type: str = 'amplitude'):
        dict_gain_type = {
            'amplitude': GainType.AMPLITUDE,
            'power': GainType.POWER,
            'db': GainType.DB,
        }
        if gain_type in dict_gain_type.keys():
            gain_type = dict_gain_type[gain_type]
        else:
            raise NotImplementedError(f"unsupported gain_type: {gain_type}")
        super(Vol, self).__init__(gain, gain_type)

    def __call__(self, waveform):
        waveform = get_numpy_data(waveform)
        return mindspore.Tensor(super(Vol, self).__call__(waveform))


class Vad(mindspore.dataset.audio.Vad):
    def __init__(self, sample_rate: int, trigger_level: float = 7.0, trigger_time: float = 0.25,
                 search_time: float = 1.0, allowed_gap: float = 0.25, pre_trigger_time: float = 0.0,
                 boot_time: float = 0.35, noise_up_time: float = 0.1, noise_down_time: float = 0.01,
                 noise_reduction_amount: float = 1.35, measure_freq: float = 20.0, measure_duration=None,
                 measure_smooth_time: float = 0.4, hp_filter_freq: float = 50.0, lp_filter_freq: float = 6000.0,
                 hp_lifter_freq: float = 150.0, lp_lifter_freq: float = 2000.0):
        super(Vad, self).__init__(sample_rate, trigger_level, trigger_time, search_time, allowed_gap, pre_trigger_time,
                                  boot_time, noise_up_time, noise_down_time, noise_reduction_amount, measure_freq,
                                  measure_duration, measure_smooth_time, hp_filter_freq, lp_filter_freq, hp_lifter_freq,
                                  lp_lifter_freq)

    def __call__(self, waveform):
        waveform = get_numpy_data(waveform)
        return mindspore.Tensor(super(Vad, self).__call__(waveform))


class GriffinLim(mindspore.dataset.audio.GriffinLim):
    def __init__(self, n_fft: int = 400, n_iter: int = 32, win_length=None, hop_length=None, window_fn=hann_window,
                 power: float = 2.0, wkwargs=None, momentum: float = 0.99, length=None, rand_init: bool = True):
        dict_window_fn = {
            hann_window: WindowType.HANN,
            bartlett_window: WindowType.BARTLETT,
            blackman_window: WindowType.BLACKMAN,
            hamming_window: WindowType.HAMMING,
        }
        if window_fn in dict_window_fn.keys():
            window = dict_window_fn[window_fn]
        else:
            raise NotImplementedError(f"unsupported window_fn: {window_fn}")
        if wkwargs is not None:
            raise NotImplementedError(f"'wkwargs' is not implemented: {wkwargs}")
        super(GriffinLim, self).__init__(n_fft, n_iter, win_length, hop_length, window, power, momentum, length,
                                         rand_init)

    def __call__(self, specgram):
        specgram = get_numpy_data(specgram)
        return mindspore.Tensor(super(GriffinLim, self).__call__(specgram))


class Fade(mindspore.dataset.audio.Fade):
    def __init__(self, fade_in_len: int = 0, fade_out_len: int = 0, fade_shape: str = 'linear'):
        dict_fade_shape = {
            'quarter_sine': FadeShape.QUARTER_SINE,
            'half_sine': FadeShape.HALF_SINE,
            'linear': FadeShape.LINEAR,
            'logarithmic': FadeShape.LOGARITHMIC,
            'exponential': FadeShape.EXPONENTIAL,
        }
        if fade_shape in dict_fade_shape.keys():
            fade_shape = dict_fade_shape[fade_shape]
        else:
            raise NotImplementedError(f"unsupported fade_shape: {fade_shape}")
        super(Fade, self).__init__(fade_in_len, fade_out_len, fade_shape)

    def __call__(self, waveform):
        waveform = get_numpy_data(waveform)
        return mindspore.Tensor(super(Fade, self).__call__(waveform))


def riaa_biquad(waveform, sample_rate: int):
    waveform = get_numpy_data(waveform)
    return mindspore.Tensor(mindspore.dataset.audio.RiaaBiquad(sample_rate)(waveform))