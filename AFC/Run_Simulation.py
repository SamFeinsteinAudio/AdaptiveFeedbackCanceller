from FeedbackCanceller.CC_FBC import CrossCorrFeedbackCanceller
from FeedbackCanceller.FDAF_FBC import FrequencyDomainFeedbackCanceller
from FeedbackCanceller.LMS_FBC import LMSFeedbackCanceller
from FeedbackSimulator.FbSim import FbSim
from Measurements.audio_calculations import calc_dbrms, calc_scalar, estimate_loudness
from SourceSignals.Noise.noise_generator import NoiseGenerator
from SourceSignals.Synth.sin_with_harmonics_generator import SinWithHarmonicsGenerator
from SourceSignals.Synth.square_generator import SquareGenerator

import random

import matplotlib.pyplot as plt
import numpy as np


SOURCES = {
    "square": SquareGenerator,
    "sin": SinWithHarmonicsGenerator,
    "noise": NoiseGenerator,
}
FB_PEDALS = {
    "CC": CrossCorrFeedbackCanceller,
    "LMS": LMSFeedbackCanceller,
    "FDAF": FrequencyDomainFeedbackCanceller,
}


def _room_trajectory(
    total_samples,
    initial_gain_db=-3.0,
    final_gain_db=3.0,
    initial_delay_ms=30.0,
    final_delay_ms=15.0,
):
    """Return a repeatable stationary (stabl), moving, then stationary (unstable) loop path."""
    section_length = total_samples // 3
    return (
        np.concatenate(
            (
                np.full(section_length, initial_gain_db),
                np.linspace(initial_gain_db, final_gain_db, section_length),
                np.full(section_length, final_gain_db),
            )
        ),
        np.concatenate(
            (
                np.full(section_length, initial_delay_ms),
                np.linspace(initial_delay_ms, final_delay_ms, section_length),
                np.full(section_length, final_delay_ms),
            )
        ),
    )


def _performance_schedule(total_samples, samplerate, seed):
    """Precompute source changes so every algorithm sees identical content."""
    rng = random.Random(seed)
    update_interval = max(1, round(0.5 * samplerate))
    scales = np.empty(total_samples)
    frequencies = np.empty(total_samples)
    scale = 1.0
    frequency = 440.0
    for sample_index in range(total_samples):
        if sample_index % update_interval == 0:
            scale = calc_scalar(rng.uniform(-18.0, -6.0))
            # Three octaves covers a realistic broad vocal range.
            frequency = 100.0 * 2 ** rng.uniform(0.0, 3.0)
        scales[sample_index] = scale
        frequencies[sample_index] = frequency
    return scales, frequencies


def _metric_summary(singer, output, feedback, samplerate, output_limit):
    residual = output - singer
    signal_level = calc_dbrms(singer)
    error_level = calc_dbrms(residual)
    return {
        "signal_level_db": signal_level,
        "residual_level_db": error_level,
        "signal_to_error_db": signal_level - error_level,
        "feedback_level_db": calc_dbrms(feedback),
        "peak_microphone_level": float(np.max(np.abs(singer + feedback))),
        "peak_output_level": float(np.max(np.abs(output))),
        "output_limit_events": int(np.count_nonzero(np.abs(output) >= output_limit * 0.999)),
        "duration_seconds": len(singer) / samplerate,
    }


def run_simulation(
    source="square",
    fb_alg="LMS",
    samplerate=44100,
    time=30,
    *,
    seed=18,
    impulse_response=None,
    initial_gain_db=-3.0,
    final_gain_db=3.0,
    initial_delay_ms=30.0,
    final_delay_ms=15.0,
    instability_threshold=8.0,
    plot=True,
    plot_mode="loudness",
    loudness_calibration=2.0,
    verbose=True,
    **pedal_kwargs,
):
    """Run one closed-loop AFC scenario and return signals plus measured metrics.

    Use ``fb_alg="none"`` for an unmodified pedal baseline. The simulation
    terminates when the microphone signal crosses ``instability_threshold`` so
    an unstable baseline does not overflow and corrupt comparison metrics.
    """
    singer_type = SOURCES.get(source)
    if singer_type is None:
        raise ValueError(f"{source} not in possible sources: {tuple(SOURCES)}")
    if fb_alg != "none" and fb_alg not in FB_PEDALS:
        raise ValueError(f"{fb_alg} not in possible algorithms: {(*FB_PEDALS, 'none')}")
    if time <= 0:
        raise ValueError("time must be positive")
    if plot_mode not in {"raw", "loudness", "spectrum"}:
        raise ValueError("plot_mode must be 'raw', 'loudness', or 'spectrum'")
    if loudness_calibration <= 0:
        raise ValueError("loudness_calibration must be positive")

    total_samples = int(round(time * samplerate))
    total_samples += (-total_samples) % 3
    room_gain_db, room_delay_ms = _room_trajectory(
        total_samples,
        initial_gain_db=initial_gain_db,
        final_gain_db=final_gain_db,
        initial_delay_ms=initial_delay_ms,
        final_delay_ms=final_delay_ms,
    )
    singer_scales, singer_frequencies = _performance_schedule(
        total_samples, samplerate, seed
    )

    singer = (
        singer_type(samplerate=samplerate, seed=seed)
        if source == "noise"
        else singer_type(samplerate=samplerate)
    )
    if fb_alg == "none":
        pedal = None
    else:
        pedal_kwargs = dict(pedal_kwargs)
        if fb_alg in {"CC", "LMS", "FDAF"}:
            pedal_kwargs.setdefault("seed", seed)
        pedal = FB_PEDALS[fb_alg](samplerate=samplerate, **pedal_kwargs)
    sound_system = FbSim(
        samplerate=samplerate, impulse_response=impulse_response
    )

    singer_full = np.zeros(total_samples)
    output_full = np.zeros(total_samples)
    feedback_full = np.zeros(total_samples)
    feedback_sample = 0.0
    stable = True
    processed_samples = total_samples

    for sample_index in range(total_samples):
        sound_system.set_delay_and_gain(
            room_gain_db[sample_index], room_delay_ms[sample_index]
        )
        singer_sample = singer.get_next_sample(
            scaling=singer_scales[sample_index], freq=singer_frequencies[sample_index]
        )
        microphone_sample = singer_sample + feedback_sample
        if not np.isfinite(microphone_sample) or abs(microphone_sample) > instability_threshold:
            stable = False
            processed_samples = sample_index
            break

        pedal_sample = (
            microphone_sample if pedal is None else pedal.process_sample(microphone_sample)
        )
        singer_full[sample_index] = singer_sample
        output_full[sample_index] = pedal_sample
        feedback_full[sample_index] = feedback_sample
        feedback_sample = sound_system.process_sample(pedal_sample)

    singer_full = singer_full[:processed_samples]
    output_full = output_full[:processed_samples]
    feedback_full = feedback_full[:processed_samples]
    output_limit = 1.0 if pedal is None else pedal.output_limit if hasattr(pedal, "output_limit") else 1.0
    metrics = _metric_summary(
        singer_full, output_full, feedback_full, samplerate, output_limit
    )
    metrics["stable"] = stable
    metrics["instability_time_seconds"] = (
        None if stable else processed_samples / samplerate
    )

    diagnostics = {}
    if pedal is not None:
        for attribute in ("estimated_lag", "estimated_gain", "last_confidence", "path_updates"):
            if hasattr(pedal, attribute):
                diagnostics[attribute] = getattr(pedal, attribute)
        if hasattr(pedal, "diagnostics"):
            diagnostics.update(pedal.diagnostics())
        if hasattr(pedal, "agc"):
            diagnostics["agc"] = pedal.agc.diagnostics()

    result = {
        "algorithm": fb_alg,
        "metrics": metrics,
        "canceller": diagnostics,
        "signals": {
            "clean": singer_full,
            "output": output_full,
            "feedback": feedback_full,
        },
        "room": {
            "gain_db": room_gain_db[:processed_samples],
            "delay_ms": room_delay_ms[:processed_samples],
            "impulse_response": sound_system.impulse_response.copy(),
        },
    }

    if verbose:
        status = "stable" if stable else f"unstable at {metrics['instability_time_seconds']:.3f}s"
        print(
            f"{fb_alg}: {status}; S/E={metrics['signal_to_error_db']:.2f} dB; "
            f"feedback={metrics['feedback_level_db']:.2f} dB; "
            f"limit events={metrics['output_limit_events']}"
        )
    if plot:
        _plot_result(
            result,
            samplerate,
            mode=plot_mode,
            loudness_calibration=loudness_calibration,
        )
    return result


def compare_algorithms(algorithms=("none", "FDAF", "LMS", "CC"), **simulation_kwargs):
    """Run identical, seed-controlled scenarios for each requested algorithm."""
    return {
        algorithm: run_simulation(fb_alg=algorithm, **simulation_kwargs)
        for algorithm in algorithms
    }


def _one_sided_spectrum(samples, samplerate, floor_db=-140.0):
    """Return a coherent-gain-normalized one-sided magnitude spectrum in dBFS."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 1 or len(samples) < 2:
        raise ValueError("spectrum analysis requires at least two one-dimensional samples")

    window = np.hanning(len(samples))
    coherent_gain = np.sum(window)
    if coherent_gain == 0:
        window = np.ones(len(samples))
        coherent_gain = len(samples)

    magnitude = np.abs(np.fft.rfft(samples * window)) / coherent_gain
    if len(samples) % 2 == 0:
        magnitude[1:-1] *= 2.0
    else:
        magnitude[1:] *= 2.0
    frequency = np.fft.rfftfreq(len(samples), d=1.0 / samplerate)
    magnitude_dbfs = 20.0 * np.log10(np.maximum(magnitude, 10 ** (floor_db / 20.0)))
    return frequency, magnitude_dbfs


def _plot_result(result, samplerate, mode="raw", loudness_calibration=2.0):
    """Plot raw sample amplitudes or Mosqito Zwicker loudness in sones."""
    signals = result["signals"]
    series = (
        ("Clean source", signals["clean"]),
        ("Feedback at microphone", signals["feedback"]),
        ("Pedal output", signals["output"]),
    )

    plt.figure(figsize=(10, 5))
    if mode == "raw":
        time_axis = np.arange(len(signals["clean"])) / samplerate
        for label, samples in series:
            plt.plot(time_axis, samples, label=label, alpha=0.8)
        plt.ylabel("Amplitude")
    elif mode == "loudness":
        for label, samples in series:
            loudness, time_axis = estimate_loudness(
                samples,
                calibration=loudness_calibration,
                samplerate=samplerate,
            )
            plt.plot(time_axis, loudness, label=label, alpha=0.8)
        plt.ylabel("Perceived loudness (sones)")
    elif mode == "spectrum":
        for label, samples in series:
            frequency, magnitude_dbfs = _one_sided_spectrum(samples, samplerate)
            # Frequency zero is omitted because a logarithmic frequency axis
            # cannot represent DC and it is not useful for feedback analysis.
            plt.semilogx(frequency[1:], magnitude_dbfs[1:], label=label, alpha=0.8)
        plt.ylabel("Magnitude (dBFS)")
        plt.xlim(left=max(1.0, samplerate / len(signals["clean"])))
    else:
        raise ValueError("mode must be 'raw', 'loudness', or 'spectrum'")

    plt.xlabel("Frequency (Hz)" if mode == "spectrum" else "Time (seconds)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    comparison = compare_algorithms(time=6, impulse_response=[1.0, 0.2, 0.1])
    for algorithm, result in comparison.items():
        print(f"{algorithm}: {result['metrics']}")
