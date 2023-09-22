import numpy as np
import matplotlib.pyplot as plt

class Index:
    def __init__(self, index):
        self.values = index

    def to_numpy(self):
        return np.array(self.values)

class CustomArray:
    def __init__(self, data, index=None, name=None):
        self.data = np.array(data)
        self.index = Index(index)
        self.name = name

    def plot(self, kind='line'):
        if kind == 'line':
            self._plot_line()
        elif kind == 'bar':
            self._plot_bar()
        # Add more plot types as needed

    def _plot_line(self):
        x = self.index if self.index is not None else np.arange(len(self.data))
        plt.plot(self.index, self.data)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title(self.name if self.name is not None else 'Custom Array Plot')
        plt.show()

    def _plot_bar(self):
        x = self.index if self.index is not None else np.arange(len(self.data))
        plt.bar(x, self.data)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title(self.name if self.name is not None else 'Custom Array Bar Plot')

    def __array__(self):
        return self.data

    def to_numpy(self):
        return self.data

# Example usage:
custom_data = np.array([0, 10, 20, 30, 40, 50])
custom_index = np.array([0, 100, 200, 300, 400, 500])
custom_name = 'MyCustomArray'

custom_series = CustomArray(custom_data, index=custom_index, name=custom_name)
plt.plot(custom_series)  # Use plot(custom_series) to create the plot

plt.show()