from scipy.signal import butter, lfilter
import numpy as np

class ButterworthHighPassFilter:
    def __init__(self, normal_cutoff, order=5):
        self.b, self.a = butter(order, normal_cutoff, btype='high', analog=False)
        # Initialize input and output delay buffers
        self.input_delay_buffer = [0] * (len(self.b))
        self.output_delay_buffer = [0] * (len(self.a))

    def apply(self, new_measurement):
        # Apply input delay
        self.input_delay_buffer.pop(0)
        self.input_delay_buffer.append(new_measurement)
        a = self.a
        b = self.b

        num = (b*np.array(self.input_delay_buffer[::-1]))/a[0]
        den = (a[1:]*np.array(self.output_delay_buffer[::-1][:-1]))/a[0]
        num=sum(num)
        den=sum(den)

        filtered_output = num-den

        # Apply output delay
        self.output_delay_buffer.pop(0)
        self.output_delay_buffer.append(filtered_output)

        return filtered_output