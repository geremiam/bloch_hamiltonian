import argparse
import numpy as np
import bloch_hamiltonian as bh
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

def test_AFM():
    dim = 2
    
    angles = np.linspace(0., 2.*np.pi, num=100)
    energies = []
    
    for angle in angles:
        r0 = Rotation.identity()
        r1 = Rotation.from_rotvec(angle * np.array([0,1,0]))
        sl_rotations = [r0, r1]
    
        spin_magnitudes = [0.5, 0.5]
    
        magsys = bh.magnonsystem_t(dim, spin_magnitudes, sl_rotations)
    
        magsys.add_coupling((0,0), 0, 1, heisen=1.)
        magsys.add_coupling((1,0), 0, 1, heisen=1.)
        magsys.add_coupling((0,1), 0, 1, heisen=1.)
        magsys.add_coupling((1,1), 0, 1, heisen=1.)
        
        energies.append(magsys.classical_energy())
    
    plt.plot(angles,energies)
    plt.show()
    
    magsys.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.prog = "bloch_hamiltonian.py"
#     parser.description = "Defines object for calculating Bloch Hamiltonians for various systems."
    parser.epilog = "Example usage: python3 test_AFM.py"
    args = parser.parse_args()
    
    test_AFM()
