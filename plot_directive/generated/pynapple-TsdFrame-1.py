import pynapple as nap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
noisy_data = np.random.rand(100) + np.sin(np.linspace(0, 2 * np.pi, 100))
tsd = nap.Tsd(t=np.arange(100), d=noisy_data)
new_tsd = tsd.decimate(down=4)
plt.plot(tsd, color="k", label="original") #doctest: +ELLIPSIS
# Expected:
## [<matplotlib.lines.Line2D object at 0x...>]
plt.plot(new_tsd, color="r", marker="o", label="decimate") #doctest: +ELLIPSIS
# Expected:
## [<matplotlib.lines.Line2D object at 0x...>]
plt.plot(tsd[::4], color="g", marker="o", label="naive downsample") #doctest: +ELLIPSIS
# Expected:
## [<matplotlib.lines.Line2D object at 0x...>]
plt.legend() #doctest: +ELLIPSIS
# Expected:
## <matplotlib.legend.Legend object at 0x...>
plt.show()
