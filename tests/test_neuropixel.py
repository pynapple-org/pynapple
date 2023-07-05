"""
@Author: Selen Calgin
@Date: 05-06-2023
@Last Modified by: Selen Calgin

Tests of neuropixel loader for 'pynapple' packagae

"""

import pynapple as nap
def test_load_session():
    data = nap.load_session(r"C:\Users\scalgi\OneDrive - McGill University\Peyrache Lab\Data\pynapple_test",
                     session_type='allensdk')
    print(data.epochs)
    print(data.spikes)
    print(data.stimulus_epochs_names)
    print(data.stimulus_epochs_block)
    print(data.optogenetic_stimulus_epochs)
    print(data.stimulus_presentations)
    print(data.stimulus_conditions)
    print(data.probes)
    print(data.channels)
def test_base_loader():
    nap.load_session(r"C:\Users\scalgi\OneDrive - McGill University\Peyrache "
                     r"Lab\Data\pynapple_test\temp\session_715093703")