import  matplotlib.pyplot as plt
import sys  
def plot_times(loading_times,computation_times,batch_sizes):
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, loading_times, marker='o', label='Loading Time (s)')
    plt.plot(batch_sizes, computation_times, marker='o', label='Computation + Communication Time (s)')
    #special color for optimal batch size
    plt.plot(batch_sizes[computation_times.index(min(computation_times))], min(computation_times), marker='o', color='red', label='Optimal Batch Size')
    plt.xlabel('Batch Size')     
    plt.ylabel('Time (seconds)')
    plt.title('Loading and Computation Times vs Batch Size')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("Usage: analys_device.py <analysis_file>")
        sys.exit(1)
    analysis_file = sys.argv[1]
    batch_sizes = []
    loading_times = []
    computation_times = []
    with open(analysis_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                loading_time = float(parts[0])
                computation_time = float(parts[1])
                batch_size = int(parts[3])
                loading_times.append(loading_time)
                computation_times.append(computation_time)
                batch_sizes.append(batch_size)
    plot_times(loading_times, computation_times, batch_sizes)


    