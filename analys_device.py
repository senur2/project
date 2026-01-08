import matplotlib.pyplot as plt
import sys

def plot_times(device_numbers,loading_times,computation_times):
    plt.figure(figsize=(10, 6))
    plt.plot(device_numbers, loading_times, marker='o', label='Loading Time (s)')
    plt.plot(device_numbers, computation_times, marker='o', label='Computation + Communication Time (s)')
    plt.xlabel('Number of Devices')     
    plt.ylabel('Time (seconds)')
    plt.title('Loading and Computation Times vs Number of Devices')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("Usage: analys_device.py <analysis_file>")
        sys.exit(1)
    analysis_file = sys.argv[1]
    device_numbers = []
    loading_times = []
    computation_times = []
    with open(analysis_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                loading_time = float(parts[0])
                computation_time = float(parts[1])
                device_number = int(parts[2])
                loading_times.append(loading_time)
                computation_times.append(computation_time)
                device_numbers.append(device_number)
    plot_times(device_numbers, loading_times, computation_times)
