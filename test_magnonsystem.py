import argparse
import numpy as np
import bloch_hamiltonian as bh
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import termcolor as tc

def compare(arr1, arr2, tol, prepend=''):
    assert tol > 0., 'The argument tol must be strictly positive'
    diff = np.amax(np.abs(arr1 - arr2))
    
    if diff<=tol:
        text = tc.colored('PASSED\tdiff = {}'.format(diff), 'green')
    else:
        text = tc.colored('FAILED\tdiff = {}'.format(diff), 'red', attrs=['bold'])
    
    print(prepend + text)
    
    return diff

def test_1D_FM(verbose=False, plot=False):
    print('\n--> 1D Heisenberg ferromagnet')
    
    dim = 1
    sl_rotations = [Rotation.identity()]
    spin_magnitudes = [1.]

    magsys = bh.magnonsystem_t(dim, spin_magnitudes, sl_rotations)

    magsys.add_coupling((1,), 0, 0, heisen=-1.)
    
    if verbose>1:
        magsys.show()
    
    classical_energy_expected = -1.0
    compare(classical_energy_expected, magsys.classical_energy(), 1.e-9, prepend='Classical energy: ')
    
    k = [0.1]
    ham, tau3 = magsys.bloch_ham(k, mode='RL')
    energy = np.sort( np.linalg.eigvals(tau3 @ ham) )
    
    energy_expected = np.array([-0.38196601+0.j,  0.38196601+0.j])
    compare(energy, energy_expected, 3.e-9, prepend='Magnon energy: ')
    
    if verbose:
        print('\nClassical energy: {}'.format(magsys.classical_energy()) )
        print('*** Energy at a single momentum ***')
        print(f'{magsys.spin_magnitudes = }')
        print(f'{k = }')
        print(f'{tau3.shape = }')
        print(f'{ham.shape = }')
        print(f'{(tau3 @ ham).shape = }')
        print(f'{energy = }')
    
    if plot:
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

def test_1D_AFM(verbose=False, plot=False):
    '''
    Schematic:
            ((0),1,0)   ((1),0,1)
    |-----0-----------1-----|-----0-----------1-----|
    '''
    print('\n--> 1D Heisenberg antiferromagnet')
    dim = 1
    
    r0 = Rotation.identity()
    r1 = Rotation.from_rotvec(np.pi * np.array([0,1,0]))
    sl_rotations = [r0, r1]

    spin_magnitudes = [1., 1.]

    magsys = bh.magnonsystem_t(dim, spin_magnitudes, sl_rotations)

#     magsys.add_field([0,1], [2.,0,0])

    magsys.add_coupling((0,), 1, 0, heisen=1.)
    magsys.add_coupling((1,), 0, 1, heisen=1.)
    
    if verbose>1:
        magsys.show()
    
    classical_energy_expected = -2.0
    compare(classical_energy_expected, magsys.classical_energy(), 1.e-9, prepend='Classical energy: ')
    
    k = [0.1]
    ham, tau3 = magsys.bloch_ham(k, mode='RL')
    energy = np.sort( np.linalg.eigvals(tau3 @ ham) )
    
    energy_expected = np.array([-0.61803399+0.j, -0.61803399+0.j,  0.61803399+0.j,  0.61803399+0.j])
    compare(energy, energy_expected, 3.e-9, prepend='Magnon energy: ')
    
    if verbose:
        print('\nClassical energy: {}'.format(magsys.classical_energy()) )
        print('*** Energy at a single momentum ***')
        print(f'{magsys.spin_magnitudes = }')
        print(f'{k = }')
        print(f'{tau3.shape = }')
        print(f'{ham.shape = }')
        print(f'{(tau3 @ ham).shape = }')
        print(f'{energy = }')
    
    if plot:
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

def test_2D_FM(verbose=False, plot=False):
    print('\n--> 2D Heisenberg ferromagnet')
    dim = 2
    # One spin per magnetic unit cell
    sl_rotations = [Rotation.identity()]
    spin_magnitudes = [1.]

    magsys = bh.magnonsystem_t(dim, spin_magnitudes, sl_rotations)

    magsys.add_coupling((1,0), 0, 0, heisen=-1.)
    magsys.add_coupling((0,1), 0, 0, heisen=-1.)
    
    if verbose>1:
        magsys.show()
    
    classical_energy_expected = -2.0
    compare(classical_energy_expected, magsys.classical_energy(), 1.e-9, prepend='Classical energy: ')
    
    k = [0.12, 0.23]
    ham, tau3 = magsys.bloch_ham(k, mode='RL')
    energy = np.sort( np.linalg.eigvals(tau3 @ ham) )
    
    energy_expected = np.array([-2.29139628+0.j,  2.29139628+0.j])
    compare(energy, energy_expected, 3.e-9, prepend='Magnon energy: ')
    
    if verbose:
        print('\nClassical energy: {}'.format(magsys.classical_energy()) )
        print('*** Energy at a single momentum ***')
        print(f'{magsys.spin_magnitudes = }')
        print(f'{k = }')
        print(f'{tau3.shape = }')
        print(f'{ham.shape = }')
        print(f'{(tau3 @ ham).shape = }')
        print(f'{energy = }')
    
    if plot:
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

def test_2D_AFM(verbose=False, plot=False):
    print('\n--> 2D Heisenberg antiferromagnet')
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
    
    if verbose>1:
        magsys.show()
    
    # Watch out! The primitive lattice vectors have been changed by the magnetic order.
    lattice_vectors = [[1, -1], [1,1]]
    
    classical_energy_expected = -4.0
    compare(classical_energy_expected, magsys.classical_energy(), 1.e-9, prepend='Classical energy: ')
    
    k = [1., 1.]
    ham, tau3 = magsys.bloch_ham(k, mode='cartesian', lattice_vectors=lattice_vectors)
    energy = np.sort( np.linalg.eigvals(tau3 @ ham) )
    
    energy_expected = np.array([-3.36588394+0.00000000e+00j, -3.36588394+4.14784945e-17j,  3.36588394-7.07612309e-16j,  3.36588394+0.00000000e+00j])
    compare(energy, energy_expected, 3.e-9, prepend='Magnon energy: ')
    
    if verbose:
        print('\nClassical energy: {}'.format(magsys.classical_energy()) )
        print('*** Energy at a single momentum ***')
        print(f'{magsys.spin_magnitudes = }')
        print(f'{k = }')
        print(f'{tau3.shape = }')
        print(f'{ham.shape = }')
        print(f'{(tau3 @ ham).shape = }')
        print(f'{energy = }')
    
    if plot:
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

def test_honeycomb_FM(verbose=False, plot=False):
    print('\n--> Honeycomb Heisenberg ferromagnet')
    dim = 2
    
    r0 = Rotation.identity()
    sl_rotations = [r0, r0]

    spin_magnitudes = [0.5, 1.]

    magsys = bh.magnonsystem_t(dim, spin_magnitudes, sl_rotations)

    magsys.add_coupling((0,0), 0, 1, heisen=-1.)
    magsys.add_coupling((1,0), 0, 1, heisen=-1.)
    magsys.add_coupling((0,1), 0, 1, heisen=-1.)
    
    if verbose>1:
        magsys.show()
    
    ####################################################################################
    lattice_vectors = [ [3./2., -np.sqrt(3.)/2.], [3./2., np.sqrt(3.)/2.] ]
    
    classical_energy_expected = -1.5
    compare(classical_energy_expected, magsys.classical_energy(), 1.e-9, prepend='Classical energy: ')
    
    k = [1.3, 1.]
    ham, tau3 = magsys.bloch_ham(k, mode='cartesian', lattice_vectors=lattice_vectors)
    energy = np.sort( np.linalg.eigvals(tau3 @ ham) )
    
    energy_expected = np.array([-3.44259936+3.61683665e-16j, -1.05740064+8.24055451e-17j,  1.05740064+9.04209162e-17j,  3.44259936+2.06013863e-17j])
    compare(energy, energy_expected, 3.e-9, prepend='Magnon energy: ')
    
    if verbose:
        print('\nClassical energy: {}'.format(magsys.classical_energy()) )
        print('*** Energy at a single momentum ***')
        print(f'{magsys.spin_magnitudes = }')
        print(f'{k = }')
        print(f'{tau3.shape = }')
        print(f'{ham.shape = }')
        print(f'{(tau3 @ ham).shape = }')
        print(f'{energy = }')
    
    if plot:
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

def test_honeycomb_FM_DM(verbose=False, plot=False):
    print('\n--> Honeycomb Heisenberg & DM ferromagnet')
    dim = 2
    
    r0 = Rotation.identity()
    sl_rotations = [r0, r0]

    spin_magnitudes = [1., 1.]

    magsys = bh.magnonsystem_t(dim, spin_magnitudes, sl_rotations)
    
    J = -1.
    magsys.add_coupling((0,0), 0, 1, heisen=J)
    magsys.add_coupling((1,0), 0, 1, heisen=J)
    magsys.add_coupling((0,1), 0, 1, heisen=J)
    
    DM = 0.1
    Jp = -0.3
    
    # The "directions" for the DM interaction have the lattice's inversion and threefold rotation
    magsys.add_coupling((-1,0), 0, 0, heisen=Jp, D=[0,0,DM])
    magsys.add_coupling((0,1),  0, 0, heisen=Jp, D=[0,0,DM])
    magsys.add_coupling((1,-1), 0, 0, heisen=Jp, D=[0,0,DM])

    magsys.add_coupling((1,0),  1, 1, heisen=Jp, D=[0,0,DM])
    magsys.add_coupling((0,-1), 1, 1, heisen=Jp, D=[0,0,DM])
    magsys.add_coupling((-1,1), 1, 1, heisen=Jp, D=[0,0,DM])
    
    if verbose>1:
        magsys.show()
    
    ####################################################################################
    lattice_vectors = [ [3./2., -np.sqrt(3.)/2.], [3./2., np.sqrt(3.)/2.] ]
    
    classical_energy_expected = -4.800000000000002
    compare(classical_energy_expected, magsys.classical_energy(), 1.e-9, prepend='Classical energy: ')
    
    k = [1.3, 1.0]
    ham, tau3 = magsys.bloch_ham(k, mode='cartesian', lattice_vectors=lattice_vectors)
    energy = np.sort( np.linalg.eigvals(tau3 @ ham) )
    
    energy_expected = np.array([-6.5316444 +8.54648265e-17j, -3.83660371+1.36579778e-16j,  3.83660371+1.36579778e-16j,  6.5316444 +8.54648265e-17j])
    compare(energy, energy_expected, 3.e-9, prepend='Magnon energy: ')
    
    if verbose:
        print('\nClassical energy: {}'.format(magsys.classical_energy()) )
        print('*** Energy at a single momentum ***')
        print(f'{magsys.spin_magnitudes = }')
        print(f'{k = }')
        print(f'{tau3.shape = }')
        print(f'{ham.shape = }')
        print(f'{(tau3 @ ham).shape = }')
        print(f'{energy = }')
    
    if plot:
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

def test_honeycomb_AFM(verbose=False, plot=False):
    print('\n--> Honeycomb Heisenberg antiferroferromagnet')
    dim = 2
    
    r0 = Rotation.identity()
    r1 = Rotation.from_rotvec(np.pi * np.array([0,1,0]))
    sl_rotations = [r0, r1]

    spin_magnitudes = [1., 1.5]

    magsys = bh.magnonsystem_t(dim, spin_magnitudes, sl_rotations)

    magsys.add_coupling((0,0), 0, 1, heisen=1.)
    magsys.add_coupling((1,0), 0, 1, heisen=1.)
    magsys.add_coupling((0,1), 0, 1, heisen=1.)
    
    if verbose>1:
        magsys.show()
    
    ####################################################################################
    lattice_vectors = [ [3./2., -np.sqrt(3.)/2.], [3./2., np.sqrt(3.)/2.] ]
    
    classical_energy_expected = -4.5
    compare(classical_energy_expected, magsys.classical_energy(), 1.e-9, prepend='Classical energy: ')
    
    k = [1., 1.3]
    ham, tau3 = magsys.bloch_ham(k, mode='cartesian', lattice_vectors=lattice_vectors)
    energy = np.sort( np.linalg.eigvals(tau3 @ ham) )
    
    energy_expected = np.array([-4.10681768+0.j, -2.60681768+0.j,  2.60681768+0.j,  4.10681768+0.j])
    compare(energy, energy_expected, 3.e-9, prepend='Magnon energy: ')
    
    if verbose:
        print('\nClassical energy: {}'.format(magsys.classical_energy()) )
        print('*** Energy at a single momentum ***')
        print(f'{magsys.spin_magnitudes = }')
        print(f'{k = }')
        print(f'{tau3.shape = }')
        print(f'{ham.shape = }')
        print(f'{(tau3 @ ham).shape = }')
        print(f'{energy = }')
    
    if plot:
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
    parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')
    args = parser.parse_args()
    np.set_printoptions(linewidth=250)
    
    verbose = args.verbose
    
    
    test_1D_FM(verbose=verbose)
    test_1D_AFM(verbose=verbose)
    test_2D_FM(verbose=verbose)
    test_2D_AFM(verbose=verbose)
    test_honeycomb_FM(verbose=verbose)
    test_honeycomb_FM_DM(verbose=verbose)
    test_honeycomb_AFM(verbose=verbose)
