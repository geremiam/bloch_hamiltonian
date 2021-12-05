import argparse
import numpy as np
import bloch_hamiltonian as bh
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def test_1D_FM():
    
    dim = 1
    sl_rotations = [Rotation.identity()]
    spin_magnitudes = [1.]

    magsys = bh.magnonsystem_t(dim, spin_magnitudes, sl_rotations)

    magsys.add_coupling((1,), 0, 0, heisen=-1.)
    
    print( 'Classical energy: {}'.format(magsys.classical_energy()) )
    
    magsys.show()
    
    print('\n*** Energy at a single momentum ***')
    k = [0.1]
    ham, tau3 = magsys.bloch_ham(k, mode='RL')
    print(f'{magsys.spin_magnitudes = }')
    print(f'{k = }')
    print(f'{tau3.shape = }')
    print(f'{ham.shape = }')
    print(f'{(tau3 @ ham).shape = }')
    energy = np.linalg.eigvals(tau3 @ ham)
    print(f'{energy = }')
    
    print('\n*** Plotting band structure ***')
    k = np.linspace(-1/2., 1/2., num=500)
    
    ham, tau3 = magsys.bloch_ham(k, 'RL')
    print(f'{ham.shape = }')
    
    energies = np.linalg.eigvals(tau3 @ ham)
    energies = np.sort(energies, axis=-1)
    
    fig, ax = plt.subplots()
    ax.plot(k, energies.real)
    ax.plot(k, energies.imag)
    plt.show()
    
    return

def test_1D_AFM_GS1():
    '''
    Schematic:
            ((0),1,0)   ((1),0,1)
    |-----0-----------1-----|-----0-----------1-----|
    '''
    dim = 1
    
    angles = np.linspace(-np.pi, np.pi, num=100)
    energies = []
    
    for angle in angles:
        r0 = Rotation.identity()
        r1 = Rotation.from_rotvec(angle * np.array([0,1,0]))
        sl_rotations = [r0, r1]
        
        spin_magnitudes = [1., 1.]
        
        magsys = bh.magnonsystem_t(dim, spin_magnitudes, sl_rotations)
        
        magsys.add_field([0,1], [0,0,1])
        
        magsys.add_coupling((0,), 1, 0, heisen=1.)
        magsys.add_coupling((1,), 0, 1, heisen=1.)
        
        energies.append(magsys.classical_energy())
    
    plt.plot(angles,energies)
    plt.show()
    
    magsys.show()
    
    return

def test_1D_AFM_GS2():
    '''
    Schematic:
            ((0),1,0)   ((1),0,1)
    |-----0-----------1-----|-----0-----------1-----|
    '''
    dim = 1
    
    angles = np.linspace(0., np.pi, num=100)
    energies = []
    
    for angle in angles:
        r0 = Rotation.from_rotvec(angle * np.array([0,1,0]))
        r1 = Rotation.from_rotvec((np.pi-angle) * np.array([0,1,0]))
        sl_rotations = [r0, r1]
        
        spin_magnitudes = [1., 1.]
        
        magsys = bh.magnonsystem_t(dim, spin_magnitudes, sl_rotations)
        
        magsys.add_field([0,1], [2.,0,0])
        
        magsys.add_coupling((0,), 1, 0, heisen=1.)
        magsys.add_coupling((1,), 0, 1, heisen=1.)
        
        energies.append(magsys.classical_energy())
    
    plt.plot(angles,energies)
    plt.show()
    
    magsys.show()
    
    return

def test_1D_AFM():
    '''
    Schematic:
            ((0),1,0)   ((1),0,1)
    |-----0-----------1-----|-----0-----------1-----|
    '''
    dim = 1
    
    r0 = Rotation.identity()
    r1 = Rotation.from_rotvec(np.pi * np.array([0,1,0]))
    sl_rotations = [r0, r1]

    spin_magnitudes = [1., 1.]

    magsys = bh.magnonsystem_t(dim, spin_magnitudes, sl_rotations)

#     magsys.add_field([0,1], [2.,0,0])

    magsys.add_coupling((0,), 1, 0, heisen=1.)
    magsys.add_coupling((1,), 0, 1, heisen=1.)
    
    magsys.show()
    
    print('\n*** Energy at a single momentum ***')
    k = [0.1]
    ham, tau3 = magsys.bloch_ham(k, mode='RL')
    print(f'{magsys.spin_magnitudes = }')
    print(f'{k = }')
    print(f'{tau3.shape = }')
    print(f'{ham.shape = }')
    print(f'{(tau3 @ ham).shape = }')
    energy = np.linalg.eigvals(tau3 @ ham)
    print(f'{energy = }')
    
    print('\n*** Plotting band structure ***')
    k = np.linspace(-1/2., 1/2., num=500)
    
    ham, tau3 = magsys.bloch_ham(k, 'RL')
    
    energies = np.linalg.eigvals(tau3 @ ham)
    energies = np.sort(energies, axis=-1)
    
    fig, ax = plt.subplots()
    ax.plot(k, energies.real)
    ax.plot(k, energies.imag)
    plt.show()
    
    return

def test_2D_FM():
    
    dim = 2
    # One spin per magnetic unit cell
    sl_rotations = [Rotation.identity()]
    spin_magnitudes = [1.]

    magsys = bh.magnonsystem_t(dim, spin_magnitudes, sl_rotations)

    magsys.add_coupling((1,0), 0, 0, heisen=-1.)
    magsys.add_coupling((0,1), 0, 0, heisen=-1.)
    
    print( 'Classical energy: {}'.format(magsys.classical_energy()) )
    
    magsys.show()
    
    print('\n*** Energy at a single momentum ***')
    k = [0.12, 0.23]
    ham, tau3 = magsys.bloch_ham(k, mode='RL')
    print(f'{magsys.spin_magnitudes = }')
    print(f'{k = }')
    print(f'{tau3.shape = }')
    print(f'{ham.shape = }')
    print(f'{(tau3 @ ham).shape = }')
    energy = np.linalg.eigvals(tau3 @ ham)
    print(f'{energy = }')
    
    print('\n*** Plotting band structure ***')
    k0 = np.linspace(-0.5, 0.5, num=200)
    k1 = np.linspace(-0.5, 0.5, num=300)
    k = np.meshgrid(k0, k1, indexing='ij')
    
    ham, tau3 = magsys.bloch_ham(k, mode='RL')
    print(f'{ham.shape = }')
    
    energies = np.linalg.eigvals(tau3 @ ham)
    energies = np.sort(energies, axis=-1)
    print(f'{energies.shape = }')
    
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    
    for i in range(energies.shape[-1]):
        ax.plot_surface(k[0], k[1], energies.real[...,i])
    plt.show()
        
    return

def test_2D_AFM_GS():
    dim = 2
    
    angles = np.linspace(-np.pi, np.pi, num=100)
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
    
    return

def test_2D_AFM():
    dim = 2
    # Two spins per magnetic unit cell
    r0 = Rotation.identity()
    r1 = Rotation.from_rotvec(np.pi * np.array([0,1,0]))
    sl_rotations = [r0, r1]

    spin_magnitudes = [1., 1.]

    magsys = bh.magnonsystem_t(dim, spin_magnitudes, sl_rotations)

    magsys.add_coupling((0,0), 0, 1, heisen=1.)
    magsys.add_coupling((1,0), 0, 1, heisen=1.)
    magsys.add_coupling((0,1), 0, 1, heisen=1.)
    magsys.add_coupling((1,1), 0, 1, heisen=1.)
    
    magsys.show()
    
    # Watch out! The primitive lattice vectors have been changed by the magnetic order.
    lattice_vectors = [[1, -1], [1,1]]
    
    print('\n*** Energy at a single momentum ***')
    k = [1., 1.]
    ham, tau3 = magsys.bloch_ham(k, mode='cartesian', lattice_vectors=lattice_vectors)
    print(f'{magsys.spin_magnitudes = }')
    print(f'{k = }')
    print(f'{tau3.shape = }')
    print(f'{ham.shape = }')
    print(f'{(tau3 @ ham).shape = }')
    energy = np.linalg.eigvals(tau3 @ ham)
    print(f'{energy = }')
    
    print('\n*** Plotting band structure ***')
    
    k0 = np.linspace(-0.5, 0.5, num=200)
    k1 = np.linspace(-0.5, 0.5, num=300)
    k = np.meshgrid(k0, k1, indexing='ij')
    
    ham, tau3 = magsys.bloch_ham(k, mode='RL')
    
    energies = np.linalg.eigvals(tau3 @ ham)
    energies = np.sort(energies, axis=-1)
    print(f'{energies.shape = }')
    
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    
    for i in range(energies.shape[-1]):
        ax.plot_surface(k[0], k[1], energies.real[...,i])
    plt.show()
    
    return

def test_honeycomb_FM():
    dim = 2
    
    r0 = Rotation.identity()
    sl_rotations = [r0, r0]

    spin_magnitudes = [1., 1.]

    magsys = bh.magnonsystem_t(dim, spin_magnitudes, sl_rotations)

    magsys.add_coupling((0,0), 0, 1, heisen=-1.)
    magsys.add_coupling((1,0), 0, 1, heisen=-1.)
    magsys.add_coupling((0,1), 0, 1, heisen=-1.)
    
    magsys.show()
    
    ####################################################################################
    lattice_vectors = [ [3./2., -np.sqrt(3.)/2.], [3./2., np.sqrt(3.)/2.] ]
    
    print('\n*** Energy at a single momentum ***')
    k = [1.3, 1.]
    ham, tau3 = magsys.bloch_ham(k, mode='cartesian', lattice_vectors=lattice_vectors)
    print(f'{magsys.spin_magnitudes = }')
    print(f'{k = }')
    print(f'{tau3.shape = }')
    print(f'{ham.shape = }')
    print(f'{(tau3 @ ham).shape = }')
    energy = np.linalg.eigvals(tau3 @ ham)
    print(f'{energy = }')
    
    print('\n*** Plotting band structure ***')
    k0 = np.linspace(-np.pi, np.pi, num=600)
    k1 = np.linspace(-np.pi, np.pi, num=600)
    k = np.meshgrid(k0, k1, indexing='ij')
    
    ham, tau3 = magsys.bloch_ham(k, mode='cartesian', lattice_vectors=lattice_vectors)
    
    energies = np.linalg.eigvals(tau3 @ ham)
    energies = np.sort(energies, axis=-1)
    print(f'{energies.shape = }')
    print(f'{np.amax(np.abs(energies.imag)) = }')
    
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    
    ax.plot_surface(k[0], k[1], energies.real[...,-1])
    ax.plot_surface(k[0], k[1], energies.real[...,-2])
    plt.show()
    
    return

def test_honeycomb_AFM():
    dim = 2
    
    r0 = Rotation.identity()
    r1 = Rotation.from_rotvec(np.pi * np.array([0,1,0]))
    sl_rotations = [r0, r1]

    spin_magnitudes = [2., 3.]

    magsys = bh.magnonsystem_t(dim, spin_magnitudes, sl_rotations)

    magsys.add_coupling((0,0), 0, 1, heisen=1.)
    magsys.add_coupling((1,0), 0, 1, heisen=1.)
    magsys.add_coupling((0,1), 0, 1, heisen=1.)
    
    magsys.show()
    
    ####################################################################################
    lattice_vectors = [ [3./2., -np.sqrt(3.)/2.], [3./2., np.sqrt(3.)/2.] ]
    
    print('\n*** Energy at a single momentum ***')
    k = [1., 1.3]
    ham, tau3 = magsys.bloch_ham(k, mode='cartesian', lattice_vectors=lattice_vectors)
    print(f'{magsys.spin_magnitudes = }')
    print(f'{k = }')
    print(f'{tau3.shape = }')
    print(f'{ham.shape = }')
    print(f'{(tau3 @ ham).shape = }')
    energy = np.linalg.eigvals(tau3 @ ham)
    print(f'{energy = }')
    
    print('\n*** Plotting band structure ***')
    k0 = np.linspace(-np.pi, np.pi, num=600)
    k1 = np.linspace(-np.pi, np.pi, num=600)
    k = np.meshgrid(k0, k1, indexing='ij')
    
    ham, tau3 = magsys.bloch_ham(k, mode='cartesian', lattice_vectors=lattice_vectors)
    
    energies = np.linalg.eigvals(tau3 @ ham)
    energies = np.sort(energies, axis=-1)
    print(f'{energies.shape = }')
    print(f'{np.amax(np.abs(energies.imag)) = }')
    
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    
    for i in range(energies.shape[-1]):
        ax.plot_surface(k[0], k[1], energies.real[...,i])
    plt.show()
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.prog = 'test_magnonsystem.py'
    parser.description = 'Tests for the class magnonsystem_t.'
    parser.epilog = 'Example usage: python3 test_magnonsystem.py'
    args = parser.parse_args()
    np.set_printoptions(linewidth=250)
    test_2D_AFM()
