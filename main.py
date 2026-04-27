import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


# a function that transform the mathlab data to python - numpy data
def mathlab_to_python(file_path):
    """
        Loads a .mat file and extracts the seismic data matrix.

        Parameters:
            file_path (str): Path to simulation_ricker.mat

        Returns:
            data (np.ndarray): Matrix of shape (num_samples, num_sensors)
        """

    mat = loadmat(file_path)

    for key in mat.keys():
        if not key.startswith("__"):
            data = mat[key]
            return data

    raise ValueError("No valid data found in the .mat file")



def add_pink_noise_snr(data, snr_db):
    """
    Adds approximate pink noise using frequency-domain 1/sqrt(f) shaping.
    """

    num_samples, num_sensors = data.shape
    pink_noise = np.zeros_like(data)

    for sensor in range(num_sensors):
        white = np.random.randn(num_samples)

        fft_white = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(num_samples)

        freqs[0] = freqs[1]  # avoid division by zero
        fft_pink = fft_white / np.sqrt(freqs)

        pink = np.fft.irfft(fft_pink, n=num_samples)
        pink = pink / (np.std(pink) + 1e-8)

        pink_noise[:, sensor] = pink

    signal_power = np.mean(data ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    pink_noise = pink_noise * np.sqrt(noise_power)

    return data + pink_noise


def compute_picking_error(picks_noisy, picks_gt):
    """
    Computes mean absolute picking error in samples.
    Ignores sensors where pick was not found.
    """

    valid = (picks_noisy >= 0) & (picks_gt >= 0)

    if np.sum(valid) == 0:
        return np.nan

    error = np.abs(picks_noisy[valid] - picks_gt[valid])

    return np.mean(error)


def evaluate_algorithm_vs_snr(
    data,
    picking_function,
    snr_values,
    noise_type="white",
    dt=1/2000,
    num_trials=10,
    **picking_kwargs
):
    """
    Evaluates picking algorithm under different SNR values.

    Returns:
        mean_errors_seconds
    """

    # GT from clean data
    picks_gt = picking_function(data, **picking_kwargs)

    mean_errors = []

    for snr in snr_values:
        trial_errors = []

        #average over num_trials noise realizations
        for _ in range(num_trials):

            if noise_type == "white":
                noisy_data = add_white_noise_snr(data, snr)
            elif noise_type == "pink":
                noisy_data = add_pink_noise_snr(data, snr)
            else:
                raise ValueError("noise_type must be 'white' or 'pink'")

            picks_noisy = picking_function(noisy_data, **picking_kwargs)

            error_samples = compute_picking_error(picks_noisy, picks_gt)
            error_seconds = error_samples * dt

            trial_errors.append(error_seconds)

        mean_errors.append(np.nanmean(trial_errors))

    return np.array(mean_errors)



def plot_seismogram(data, dt=1/2000, scale=0.5):
    num_samples, num_sensors = data.shape

    if num_samples < num_sensors:
        data = data.T
        num_samples, num_sensors = data.shape

    max_traces = 64
    if num_sensors > max_traces:
        step = num_sensors // max_traces
        data = data[:, ::step]
        num_samples, num_sensors = data.shape

# create the timeline
    time = np.arange(num_samples) * dt


    plt.figure(figsize=(8, 6))

    for i in range(num_sensors):
        trace = data[:, i]

        max_val = np.max(np.abs(trace))
        if max_val > 0:
            trace = trace / max_val

        x = i + trace * scale

        plt.plot(x, time, color='black', linewidth=0.8)
        plt.fill_betweenx(time, i, x, where=(x > i), color='black')

    plt.gca().invert_yaxis()
    plt.xlabel("Trace Number")
    plt.ylabel("Time (s)")
    plt.title("Seismogram")

    plt.tight_layout()
    plt.show()

# Seismogram  + picks
def plot_with_picks(data, picks, dt=1/2000, scale=0.5):
    """
       Plots a seismic wiggle plot (seismogram) from multi-sensor data.

       Each sensor trace is normalized and horizontally shifted to create
       a standard seismogram visualization. Positive amplitudes are filled
       for better visual contrast.

       Parameters:
           data (np.ndarray):
               Input seismic data of shape (num_samples, num_sensors).
               If the input is transposed, it will be corrected automatically.

           dt (float, optional):
               Sampling interval in seconds. Default is 1/2000.

           scale (float, optional):
               Scaling factor for trace amplitudes to control horizontal spread.
               Default is 0.5.

       Notes:
           - If the number of sensors is large, the function downsamples the
             traces to a maximum of 64 for visualization clarity.
           - Each trace is normalized independently to its maximum absolute value.
           - The time axis is plotted vertically and inverted, following
             standard seismic convention (time increases downward).
       """

    num_samples, num_sensors = data.shape

    if num_samples < num_sensors:
        data = data.T
        num_samples, num_sensors = data.shape

    max_traces = 64
    if num_sensors > max_traces:
        step = num_sensors // max_traces
        data = data[:, ::step]
        num_samples, num_sensors = data.shape

    # create the timeline
    time = np.arange(num_samples) * dt


    plt.figure(figsize=(8, 6))

    for i in range(num_sensors):

        trace = data[:, i]
        trace = trace / (np.max(np.abs(trace)) + 1e-8)

        x = i + trace * scale

        plt.plot(x, time, color='black', linewidth=0.8)
        plt.fill_betweenx(time, i, x, where=(x > i), color='black')

    #  הוספת נקודות picking
    valid = picks >= 0
    plt.scatter(
        np.arange(num_sensors)[valid],
        picks[valid] * dt,
        color='blue',  # 🔵 notable color
        s=30,  # big
        zorder=5,  # above the graph
        marker='x'
    )

    plt.plot(
        np.arange(num_sensors)[valid],
        picks[valid] * dt,
        color='blue',
        linewidth=2
    )

    plt.gca().invert_yaxis()
    plt.xlabel("Sensor")
    plt.ylabel("Time (s)")
    plt.title("Seismogram with Picks")

    plt.show()

def plot_with_multiple_picks(data, picks_per_sensor, dt=1/2000, scale=0.5):
    num_samples, num_sensors = data.shape
    time = np.arange(num_samples) * dt

    plt.figure(figsize=(10, 6))

    for i in range(num_sensors):
        trace = data[:, i]
        trace = trace / (np.max(np.abs(trace)) + 1e-8)

        x = i + trace * scale

        plt.plot(x, time, color='black', linewidth=0.8)
        plt.fill_betweenx(time, i, x, where=(x > i), color='black')

    # drwaing the picks
    for sensor, picks in enumerate(picks_per_sensor):
        if len(picks) > 0:
            times = np.array(picks) * dt
            sensors = np.full(len(picks), sensor)

            plt.scatter(
                sensors,
                times,
                color='blue',
                s=50,
                marker='x',
                zorder=10
            )

    plt.gca().invert_yaxis()
    plt.xlabel("Sensor")
    plt.ylabel("Time (s)")
    plt.title("Seismogram with Multiple Picks (Bonus)")
    plt.tight_layout()
    plt.show()

# a function that returns the dimension of the matrix
def get_dimension(matrix):
    return matrix.shape


def add_white_noise_snr(data, snr_db):
    """
    Adds white Gaussian noise to data based on desired SNR (in dB)

    Parameters:
        data: np.ndarray (num_samples, num_sensors)
        snr_db: desired SNR in dB

    Returns:
        noisy_data: data + noise
    """

    # Compute signal power
    signal_power = np.mean(data**2)

    # Convert SNR from dB to linear scale
    snr_linear = 10**(snr_db / 10)

    # Compute required noise power
    noise_power = signal_power / snr_linear

    # Generate white Gaussian noise
    noise = np.random.normal(
        loc=0,
        scale=np.sqrt(noise_power),
        size=data.shape
    )
    # Add noise to signal

    noisy_data = data + noise

    return noisy_data


def causal_moving_average(data, window_size):
    num_samples, num_sensors = data.shape
    avg_data = np.zeros_like(data)

    for sensor in range(num_sensors):
        signal = data[:, sensor]

        #cumulative sum - most efficient way to average
        cumsum = np.cumsum(signal)

        for t in range(num_samples):
            #calculating casual averge
            if t < window_size:
                avg_data[t, sensor] = cumsum[t] / (t + 1)
            else:
                avg_data[t, sensor] = (cumsum[t] - cumsum[t - window_size]) / window_size

    return avg_data


def causal_moving_median(data, window_size):
    num_samples, num_sensors = data.shape
    med_data = np.zeros_like(data)

    for sensor in range(num_sensors):
        for t in range(num_samples):
            #the window only looks at samples from the past, so it can run online
            start = max(0, t - window_size + 1)
            med_data[t, sensor] = np.median(data[start:t+1, sensor])

    return med_data


def increasing_percentage(signal, t, window_size):
    """
    Calculates percentage of increasing steps in a causal window ending at time t.
    """

    start = max(0, t - window_size + 1)
    window = signal[start:t + 1]

    if len(window) < 2:
        return 0.0,0.0

    diffs = np.diff(window)

    percent_increasing = np.sum(diffs > 0) / len(diffs)
    #how much we increase
    trend_strength = window[-1] - window[0]


    return percent_increasing, trend_strength


def picking_mean_median_causal(data, Na=11, Nm=11, thresholda=0.01, thresholdm=0.01):
    """
        Picking based on casual moving average + casual moving median
    """

    num_samples, num_sensors = data.shape

#to detect both positive and negative pulses
    data_proc = np.abs(data)

    # to make the data within the range of [0, 1] so we can choose the threshold
    data_proc = data_proc / (np.max(data_proc) + 1e-8)

    # casual calculation
    avg_data = causal_moving_average(data_proc, Na)
    med_data = causal_moving_median(data_proc, Nm)

    picks = np.full(num_sensors, -1)

    for sensor in range(num_sensors):
        for t in range(num_samples):

            cond_avg = avg_data[t, sensor] > thresholda
            cond_med = med_data[t, sensor] > thresholdm

            if cond_avg and cond_med:
                picks[sensor] = t
                break

    return picks

def picking_continuous_increasing(
    data,
    Na=21,
    Nm=21,
    Wa=20,
    Wm=20,
    percent_threshold_a=0.7,
    percent_threshold_m=0.7,
    amplitude_threshold_a=0.02,
    amplitude_threshold_m=0.02

):
    """
    Picking algorithm for continuous increasing signal.

    Parameters:
        data: np.ndarray
            Matrix of shape (num_samples, num_sensors)

        Na:
            Window size for causal moving average

        Nm:
            Window size for causal moving median

        Wa:
            Window size for checking increasing behavior of average

        Wm:
            Window size for checking increasing behavior of median

        percent_threshold_a:
            Required percentage of increasing steps for average

        percent_threshold_m:
            Required percentage of increasing steps for median

       amplitude_threshold_a (float):
        Minimum required total increase in the moving average window.

       amplitude_threshold_m (float):
        Minimum required total increase in the moving median window.

    Returns:
        picks:
            Array of pick sample index for each sensor
    """

    num_samples, num_sensors = data.shape

    #to handle both positive and negative signals
    data_proc = np.abs(data)
    #normalizing the data
    data_proc = data_proc / (np.max(data_proc) + 1e-8)

    avg_data = causal_moving_average(data_proc, Na)
    med_data = causal_moving_median(data_proc, Nm)

    picks = np.full(num_sensors, -1)

    for sensor in range(num_sensors):
        avg_signal = avg_data[:, sensor]
        med_signal = med_data[:, sensor]

        for t in range(num_samples):
            #calculating increasing percentage on the average/medium signal to reduce impact of noise
            perc_avg, amp_a = increasing_percentage(avg_signal, t, Wa)
            perc_med, amp_m = increasing_percentage(med_signal, t, Wm)

            cond_avg = perc_avg > percent_threshold_a and (amp_a > amplitude_threshold_a)
            cond_med = perc_med > percent_threshold_m and (amp_m > amplitude_threshold_m)

            if cond_avg or cond_med:
                picks[sensor] = t
                break

    return picks
def windowed_picking_per_sensor(
    data,
    window_size,
    picking_function,
    min_pick_distance=10,
    **picking_kwargs
):
    num_samples, num_sensors = data.shape
    picks_per_sensor = []

    for sensor in range(num_sensors):

        signal = data[:, sensor]
        sensor_picks = []

        for start in range(0, num_samples, window_size):

            end = min(start + window_size, num_samples)
            window = signal[start:end]

            # adjast the format to my algorithm format
            window_2d = window.reshape(-1, 1)

            picks = picking_function(window_2d, **picking_kwargs)
            pick_local = picks[0]

            # to prevent the pick to be at the beginning of the window as part of previous window
            if pick_local >= min_pick_distance:
                pick_global = start + pick_local
                sensor_picks.append(pick_global)

        picks_per_sensor.append(sensor_picks)

    return picks_per_sensor


# 🚀 MAIN
# =====================

if __name__ == "__main__":

#load files
    file_path1 = r"C:\Users\adian\Downloads\simulation_ricker (1).mat"
    file_path2 =  r"C:\Users\adian\Downloads\simulation_continuous.mat"
    file_path3 = r"C:\Users\adian\Downloads\simulation_multisource.mat"


#convert them to python files
    data1 = mathlab_to_python(file_path2)
    data2 = mathlab_to_python(file_path2)
    data3 = mathlab_to_python(file_path3)


    #  dimension correction if required

    if data1.shape[0] < data1.shape[1]:
        data1 = data1.T

    if data2.shape[0] < data2.shape[1]:
        data2 = data2.T

    if data3.shape[0] < data3.shape[1]:
        data3 = data3.T

    #signlas histogram plot
    plot_seismogram(data1,dt = 1/2000, scale=0.5)
    plot_seismogram(data2, dt=1 / 2000, scale=0.5)
    plot_seismogram(data3,dt = 1/2000, scale=0.5)





# =====================
#  Regular signal testing
# =====================
    picks = picking_mean_median_causal(
        data1,
        Na=11,
        Nm=11,
        thresholda=0.05,
        thresholdm=0.05
    )



    # Conversion to time
    dt = 1 / 2000
    pick_times = picks * dt

    #Drawing
    plot_with_picks(data1, picks, dt)






# =====================
#  Increasing signal testing
# =====================


    if data2.shape[0] < data2.shape[1]:
        data2 = data2.T

    picks = picking_continuous_increasing(
        data2,
        Na=21,
        Nm=21,
        Wa=30,
        Wm=30,
        percent_threshold_a=0.8,
        percent_threshold_m=0.8,
        amplitude_threshold_a=0.01,
        amplitude_threshold_m=0.01

    )

    dt = 1 / 2000
    pick_times = np.where(picks >= 0, picks * dt, np.nan)
    plot_with_picks(data2, picks, dt=dt, scale=0.5)


# =====================
#  SNR testing with regular picking algorithm
# =====================



    snr_values = [30, 25, 20, 15, 12, 10, 8, 5, 2, 0]
    # GT = picks on clean data
    picks_gt = picking_mean_median_causal(
        data1,
        Na=11,
        Nm=11,
        thresholda=0.07,
        thresholdm=0.07
    )


    # Plot GT picks on clean data
    plot_with_picks(data1, picks_gt, dt=dt, scale=0.5)


    dt = 1 / 2000

    #white noise
    white_errors = evaluate_algorithm_vs_snr(
        data1,
        picking_mean_median_causal,
        snr_values,
        noise_type="white",
        dt=dt,
        num_trials=10,
        Na=11,
        Nm=11,
        thresholda=0.07,
        thresholdm=0.07
    )

    #pink noise
    pink_errors = evaluate_algorithm_vs_snr(
        data1,
        picking_mean_median_causal,
        snr_values,
        noise_type="pink",
        dt=dt,
        num_trials=10,
        Na=11,
        Nm=11,
        thresholda=0.07,
        thresholdm=0.07
    )
    plt.figure(figsize=(8, 5))

    plt.plot(snr_values, white_errors * 1000, marker='o', label="White noise")
    plt.plot(snr_values, pink_errors * 1000, marker='o', label="Pink noise")

    plt.gca().invert_xaxis()
    plt.xlabel("SNR (dB)")
    plt.ylabel("Mean picking error (ms)")
    plt.title("Picking Error vs SNR")
    plt.grid(True)
    plt.legend()
    plt.show()

# =====================
# 🎯 BONUS
# =====================

picks_all = windowed_picking_per_sensor(
    data3,
    window_size=100,
    picking_function=picking_mean_median_causal,
    Na=11,
    Nm=11,
    thresholda=0.05,
    thresholdm=0.05
)



plot_with_multiple_picks(data3, picks_all, dt=1 / 2000)









