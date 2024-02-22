"""Test configuration script."""
import numpy as np



class MockArray:
    """
    A mock array class designed for testing purposes. It mimics the behavior of array-like objects
    by providing necessary attributes and supporting indexing and iteration, but it is not a direct
    instance of numpy.ndarray.
    """

    def __init__(self, data):
        """
        Initializes the MockArray with data.
        Parameters
        ----------
        data : Union[numpy.ndarray, List]
            A list of data elements that the MockArray will contain.
        """
        self.data = np.asarray(data)
        self.shape = self.data.shape # Simplified shape attribute
        self.dtype = 'float64'  # Simplified dtype; in real scenarios, this should be more dynamic
        self.ndim = self.data.ndim  # Simplified ndim for a 1-dimensional array

    def __getitem__(self, index):
        """
        Supports indexing into the mock array.
        Parameters
        ----------
        index : int or slice
            The index or slice of the data to access.
        Returns
        -------
        The element(s) at the specified index.
        """
        return self.data[index]

    def __iter__(self):
        """
        Supports iteration over the mock array.
        """
        return iter(self.data)

    def __len__(self):
        """
        Returns the length of the mock array.
        """
        return len(self.data)