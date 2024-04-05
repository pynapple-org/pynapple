import pynapple as nap
import numpy as np
import pandas as pd

n_unit = 100
df = pd.DataFrame()
df["isi_violations_ratio"] = np.random.uniform(0, 0.7, size=n_unit)
df["firing_rate"] = np.random.uniform(0, 80, size=n_unit)
df["presence_ratio"] = np.random.uniform(0.7, 1., size=n_unit)
df["device_name"] = [f"Probe{k%4}" for k in range(n_unit)]



dd = {k: np.sort(np.random.uniform(size=n_unit)) for k in range(n_unit)}
units = nap.TsGroup(dd)
units.set_info(df)

pass_qc = units[(units.isi_violations_ratio < 0.5) &
                (units.firing_rate > 0.1) &
                (units.presence_ratio > 0.95) &
                (units.device_name == 'Probe1')]
#pass_qc["new"] = np.arange(len(pass_qc))
pass_qc[1.1] = np.arange(len(pass_qc)) + 10