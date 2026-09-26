#include "mtmd-audio.h"

#define _USE_MATH_DEFINES // for M_PI
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>
#include <fstream>
#include <algorithm>
#include <functional>

// some of the code here is copied from whisper.cpp

constexpr bool DEBUG = false;

void mtmd_audio_cache::fill_sin_cos_table(uint32_t n) {
    sin_vals.resize(n);
    cos_vals.resize(n);
    for (uint32_t i = 0; i < n; i++) {
        double theta = (2 * M_PI * i) / n;
        sin_vals[i]  = sinf(theta);
        cos_vals[i]  = cosf(theta);
    }
}

void mtmd_audio_cache::fill_hann_window(uint32_t length, bool periodic) {
    hann_window.resize(length);
    int offset = periodic ? 0 : -1;
    for (uint32_t i = 0; i < length; i++) {
        hann_window[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
    }
}

void mtmd_audio_cache::fill_mel_filterbank_matrix(int64_t n_mel,
                                                  int64_t n_fft,
                                                  int   sample_rate,
                                                  float fmin,
                                                  float fmax,
                                                  bool  slaney_area_norm,
                                                  float scale,
                                                  bool  use_htk) {
    GGML_ASSERT(n_mel > 0 && n_fft > 1);
    if (fmax <= 0.0f) {
        fmax = 0.5f * sample_rate;
    }

    std::function<double(double)> hz_to_mel;
    std::function<double(double)> mel_to_hz;

    if (use_htk) {
        hz_to_mel = [](const double f_hz) -> double {
            return 2595.0 * log10(1.0 + f_hz / 700.0);
        };
        mel_to_hz = [](const double m) -> double {
            return 700.0 * (pow(10.0, m / 2595.0) - 1.0);
        };
    } else {
        // Slaney scale (matches librosa default)
        const double min_log_hz  = 1000.0;
        const double lin_slope   = 3 / 200.;
        const double min_log_mel = min_log_hz * lin_slope;
        const double log_step    = log(6.4) / 27.0;
        hz_to_mel = [min_log_hz, lin_slope, log_step, min_log_mel](const double f_hz) -> double {
            return (f_hz < min_log_hz) ? f_hz * lin_slope : min_log_mel + log(f_hz / min_log_hz) / log_step;
        };
        mel_to_hz = [min_log_hz, lin_slope, log_step, min_log_mel](const double m) -> double {
            return (m < min_log_mel) ? m / lin_slope : min_log_hz * exp((m - min_log_mel) * log_step);
        };
    }

    // infer N_fft from n_fft_bins
    const double bin_hz_step = double(sample_rate) / double(n_fft);

    // mel grid: n_mel + 2 edges
    const double        m_lo = hz_to_mel(fmin);
    const double        m_hi = hz_to_mel(fmax);
    std::vector<double> mel_pts(n_mel + 2);
    for (int i = 0; i < n_mel + 2; ++i) {
        mel_pts[i] = m_lo + (m_hi - m_lo) * (double(i) / (n_mel + 1));
    }

    // convert to Hz
    std::vector<double> hz_pts(n_mel + 2);
    for (int i = 0; i < n_mel + 2; ++i) {
        hz_pts[i] = mel_to_hz(mel_pts[i]);
    }

    const int64_t n_fft_bins = n_fft / 2 + 1;

    // Validate allocation size
    if ((size_t)n_mel * (size_t)n_fft_bins > SIZE_MAX) {
        GGML_ASSERT(false && "mel filterbank allocation too large");
    }

    // filterbank
    std::vector<float> out((size_t)n_mel * (size_t)n_fft_bins, 0);
    for (int64_t m = 0; m < n_mel; ++m) {
        const double f_left   = hz_pts[m];
        const double f_center = hz_pts[m + 1];
        const double f_right  = hz_pts[m + 2];

        const double denom_l = std::max(1e-30, f_center - f_left);
        const double denom_r = std::max(1e-30, f_right - f_center);
        const double enorm   = slaney_area_norm ? (2.0 / std::max(1e-30, f_right - f_left)) : 1.0;

        for (int k = 0; k < n_fft_bins; ++k) {
            const double f = k * bin_hz_step;
            double       w = 0.0;
            if (f >= f_left && f <= f_center) {
                w = (f - f_left) / denom_l;
            } else if (f > f_center && f <= f_right) {
                w = (f_right - f) / denom_r;
            }
            out[size_t(m) * size_t(n_fft_bins) + size_t(k)] = float(w * enorm * scale);
        }
    }

    filters.n_mel = n_mel;
    filters.n_fft = n_fft;
    filters.data  = std::move(out);

    if (DEBUG) {  // debug
        for (size_t i = 0; i < filters.data.size(); ++i) {
            if (filters.data[i] != 0.0f) {
                printf("filters[%zu] = %f\n", i, filters.data[i] * 1000.0f);
            }
        }
    }
}

// Unified DFT implementation for both forward and inverse transforms
// Template parameters:
//   Inverse: false = DFT with exp(-2πi·k·n/N), no scaling
//            true  = IDFT with exp(+2πi·k·n/N), scales by 1/N
//   RealInput: true = input is real-valued (stride 1), avoids imaginary computations
//              false = input is complex-valued (interleaved real/imag, stride 2)
template <bool Inverse, bool RealInput>
static void dft_impl(const mtmd_audio_cache & cache, const float * in, int N, float * out) {
    const int n_sin_cos_vals = cache.sin_vals.size();
    const int sin_cos_step   = n_sin_cos_vals / N;

    constexpr float sign  = Inverse ? 1.0f : -1.0f;
    const float     scale = Inverse ? (1.0f / N) : 1.0f;

    for (int k = 0; k < N; k++) {
        float re = 0;
        float im = 0;

        for (int n = 0; n < N; n++) {
            int   idx     = (k * n * sin_cos_step) % n_sin_cos_vals;
            float cos_val = cache.cos_vals[idx];
            float sin_val = cache.sin_vals[idx];

            if constexpr (RealInput) {
                // Real input: in_im = 0, simplifies to:
                // re += in_re * cos_val
                // im += sign * in_re * sin_val
                float in_re = in[n];
                re += in_re * cos_val;
                im += sign * in_re * sin_val;
            } else {
                float in_re = in[n * 2 + 0];
                float in_im = in[n * 2 + 1];
                // (a + bi) * (cos + sign*i*sin) = (a*cos - sign*b*sin) + (sign*a*sin + b*cos)i
                re += in_re * cos_val - sign * in_im * sin_val;
                im += sign * in_re * sin_val + in_im * cos_val;
            }
        }

        out[k * 2 + 0] = re * scale;
        out[k * 2 + 1] = im * scale;
    }
}

// Cooley-Tukey FFT/IFFT unified implementation
// Template parameters:
//   Inverse: false = FFT with exp(-2πi·k/N), no scaling
//            true  = IFFT with exp(+2πi·k/N), scales by 0.5 at each level
//   RealInput: true = input is real-valued (stride 1)
//              false = input is complex-valued (interleaved real/imag, stride 2)
template <bool Inverse, bool RealInput>
static void fft_impl(const mtmd_audio_cache & cache, float * in, int N, float * out) {
    GGML_ASSERT(N > 0);
    const int n_sin_cos_vals = cache.sin_vals.size();

    if (N == 1) {
        out[0] = in[0];
        if constexpr (RealInput) {
            out[1] = 0.0f;
        } else {
            out[1] = in[1];
        }
        return;
    }

    const int half_N = N / 2;
    if (N - half_N * 2 == 1) {
        // Odd N: fall back to DFT
        dft_impl<Inverse, RealInput>(cache, in, N, out);
        return;
    }

    // Split into even and odd
    if constexpr (RealInput) {
        // Real input: stride is 1, copy only real values
        float * even = in + N;
        for (int i = 0; i < half_N; ++i) {
            even[i] = in[2 * i];
        }
        float * even_fft = out + 2 * N;
        fft_impl<Inverse, true>(cache, even, half_N, even_fft);

        float * odd = even;
        for (int i = 0; i < half_N; ++i) {
            odd[i] = in[2 * i + 1];
        }
        float * odd_fft = even_fft + N;
        fft_impl<Inverse, true>(cache, odd, half_N, odd_fft);
    } else {
        // Complex input: stride is 2, copy complex pairs
        float * even = in + N * 2;
        for (int i = 0; i < half_N; ++i) {
            even[i * 2 + 0] = in[2 * i * 2 + 0];
            even[i * 2 + 1] = in[2 * i * 2 + 1];
        }
        float * even_fft = out + 2 * N;
        fft_impl<Inverse, false>(cache, even, half_N, even_fft);

        float * odd = even;
        for (int i = 0; i < half_N; ++i) {
            odd[i * 2 + 0] = in[(2 * i + 1) * 2 + 0];
            odd[i * 2 + 1] = in[(2 * i + 1) * 2 + 1];
        }
        float * odd_fft = even_fft + N;
        fft_impl<Inverse, false>(cache, odd, half_N, odd_fft);
    }

    float * even_fft = out + 2 * N;
    float * odd_fft  = even_fft + N;

    const int sin_cos_step = n_sin_cos_vals / N;

    constexpr float sign  = Inverse ? 1.0f : -1.0f;
    constexpr float scale = Inverse ? 0.5f : 1.0f;

    for (int k = 0; k < half_N; k++) {
        int   idx = k * sin_cos_step;  // t = 2*M_PI*k/N
        float re  = cache.cos_vals[idx];
        float im  = sign * cache.sin_vals[idx];

        float re_odd = odd_fft[2 * k + 0];
        float im_odd = odd_fft[2 * k + 1];

        out[2 * k + 0] = scale * (even_fft[2 * k + 0] + re * re_odd - im * im_odd);
        out[2 * k + 1] = scale * (even_fft[2 * k + 1] + re * im_odd + im * re_odd);

        out[2 * (k + half_N) + 0] = scale * (even_fft[2 * k + 0] - re * re_odd + im * im_odd);
        out[2 * (k + half_N) + 1] = scale * (even_fft[2 * k + 1] - re * im_odd - im * re_odd);
    }
}

// Forward FFT for real input (used by mel spectrogram)
static void fft(const mtmd_audio_cache & cache, float * in, int N, float * out) {
    fft_impl<false, true>(cache, in, N, out);
}

// Inverse FFT for complex input
static void ifft(const mtmd_audio_cache & cache, float * in, int N, float * out) {
    fft_impl<true, false>(cache, in, N, out);
}

//
// mtmd_audio_streaming_istft implementation
//

mtmd_audio_streaming_istft::mtmd_audio_streaming_istft(int n_fft, int hop_length) :
    n_fft(n_fft),
    hop_length(hop_length),
    n_fft_bins(n_fft / 2 + 1),
    overlap_buffer(n_fft, 0.0f),
    window_sum_buffer(n_fft, 0.0f),
    padding_to_remove((n_fft - hop_length) / 2),
    ifft_in(n_fft * 2 * 4, 0.0f),  // extra space for recursive IFFT
    ifft_out(n_fft * 2 * 4, 0.0f) {
    GGML_ASSERT(n_fft > 0 && hop_length > 0 && hop_length <= n_fft);
    cache.fill_sin_cos_table(n_fft);
    cache.fill_hann_window(n_fft, true);
}

void mtmd_audio_streaming_istft::reset() {
    std::fill(overlap_buffer.begin(), overlap_buffer.end(), 0.0f);
    std::fill(window_sum_buffer.begin(), window_sum_buffer.end(), 0.0f);
    padding_to_remove = (n_fft - hop_length) / 2;
}

std::vector<float> mtmd_audio_streaming_istft::process_frame(const float * frame_spectrum) {
    std::vector<float> output(hop_length);

    // copy frequencies
    for (int j = 0; j < n_fft_bins; j++) {
        ifft_in[j * 2 + 0] = frame_spectrum[j * 2 + 0];
        ifft_in[j * 2 + 1] = frame_spectrum[j * 2 + 1];
    }

    // mirror negative frequencies
    for (int j = 1; j < n_fft_bins - 1; j++) {
        int mirror_idx              = n_fft - j;
        ifft_in[mirror_idx * 2 + 0] = ifft_in[j * 2 + 0];
        ifft_in[mirror_idx * 2 + 1] = -ifft_in[j * 2 + 1];  // conjugate
    }

    ifft(cache, ifft_in.data(), n_fft, ifft_out.data());

    // update window sum and overlap buffer
    for (int j = 0; j < n_fft; j++) {
        window_sum_buffer[j] += cache.hann_window[j] * cache.hann_window[j];
        overlap_buffer[j] += ifft_out[j * 2] * cache.hann_window[j];
    }

    // extract hop_length samples with normalization
    for (int i = 0; i < hop_length; i++) {
        if (window_sum_buffer[i] > 1e-8f) {
            output[i] = overlap_buffer[i] / window_sum_buffer[i];
        } else {
            output[i] = overlap_buffer[i];
        }
    }

    // shift buffers left by hop_length
    std::copy(overlap_buffer.begin() + hop_length, overlap_buffer.end(), overlap_buffer.begin());
    std::fill(overlap_buffer.end() - hop_length, overlap_buffer.end(), 0.0f);

    std::copy(window_sum_buffer.begin() + hop_length, window_sum_buffer.end(), window_sum_buffer.begin());
    std::fill(window_sum_buffer.end() - hop_length, window_sum_buffer.end(), 0.0f);

    // Remove padding if needed
    int to_remove = std::min(padding_to_remove, (int) output.size());
    padding_to_remove -= to_remove;
    output.erase(output.begin(), output.begin() + to_remove);

    return output;
}

std::vector<float> mtmd_audio_streaming_istft::flush() {
    std::vector<float> output;

    // Extract remaining samples from overlap buffer
    // Continue until we've extracted all meaningful samples
    int remaining = n_fft - hop_length;
    while (remaining > 0) {
        int chunk_size = std::min(remaining, hop_length);

        for (int i = 0; i < chunk_size; i++) {
            float sample;
            if (window_sum_buffer[i] > 1e-8f) {
                sample = overlap_buffer[i] / window_sum_buffer[i];
            } else {
                sample = overlap_buffer[i];
            }
            output.push_back(sample);
        }

        // Shift buffers
        std::copy(overlap_buffer.begin() + chunk_size, overlap_buffer.end(), overlap_buffer.begin());
        std::fill(overlap_buffer.end() - chunk_size, overlap_buffer.end(), 0.0f);

        std::copy(window_sum_buffer.begin() + chunk_size, window_sum_buffer.end(), window_sum_buffer.begin());
        std::fill(window_sum_buffer.end() - chunk_size, window_sum_buffer.end(), 0.0f);

        remaining -= chunk_size;
    }

    return output;
}

