import argparse
import numpy as np
import bloch_hamiltonian as bh
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def XXZ_2D():
    
    dim = 2
    
    r0 = Rotation.from_rotvec(np.pi/2. * np.array([0,1,0]))
    sl_rotations = [r0, r0]

    spin_magnitudes = [1., 1.]

    magsys = bh.magnonsystem_t(dim, spin_magnitudes, sl_rotations, verbose=True)
    
    J1 = -1.
    alpha1 = 0.5
    magsys.add_coupling((0,0), 0, 1, Jdiag=[J1, J1, alpha1*J1])
    magsys.add_coupling((1,0), 0, 1, Jdiag=[J1, J1, alpha1*J1])
    magsys.add_coupling((0,1), 0, 1, Jdiag=[J1, J1, alpha1*J1])
    
    DM = [0.35, 0., 0.]
    
    # The "directions" for the DM interaction have the lattice's inversion and threefold rotation
    C3 = Rotation.from_rotvec(2.*np.pi/3. * np.array([0,0,1]))
    magsys.add_coupling((1,-1), 0, 0, D=DM)
    magsys.add_coupling((0,1),  0, 0, D=C3.apply(DM, inverse=False))
    magsys.add_coupling((-1,0), 0, 0, D=C3.apply(DM, inverse=True))
    
    magsys.add_coupling((-1,1), 1, 1, D=DM)
    magsys.add_coupling((0,-1), 1, 1, D=C3.apply(DM, inverse=False))
    magsys.add_coupling((1,0),  1, 1, D=C3.apply(DM, inverse=True))
    
    magsys.show()
    
    ####################################################################################
    lattice_vectors = [ [3./2., -np.sqrt(3.)/2.], [3./2., np.sqrt(3.)/2.] ]
    
    print('\n*** Energy at a single momentum ***')
    k = [1.3, 1.0]
    ham, tau3 = magsys.bloch_ham(k, mode='cartesian', lattice_vectors=lattice_vectors)
    print(f'{magsys.spin_magnitudes = }')
    print(f'{k = }')
    print(f'{tau3.shape = }')
    print('ham =\n{}'.format(ham))
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

def XXZ_3D_setinteractions(magsys, params):
    '''
    Take magnonsystem_t instance and set interactions according to values in "params".
    '''
    
    # Zeeman fields
    H_Ea = params['H_Ea']
    H_Eb = params['H_Eb']
    H_Oa = params['H_Oa']
    H_Ob = params['H_Ob']
    magsys.add_field(0, H_Ea * magsys.spin_dir(0))
    magsys.add_field(1, H_Eb * magsys.spin_dir(1))
    magsys.add_field(2, H_Oa * magsys.spin_dir(2))
    magsys.add_field(3, H_Ob * magsys.spin_dir(3))
    
    # Interactions
    J1E  = - params['J1E']
    alpha1E  = params['alpha1E']
    magsys.add_coupling((0,0,0), 0, 1, Jdiag=[J1E, J1E, alpha1E*J1E])
    magsys.add_coupling((1,0,0), 0, 1, Jdiag=[J1E, J1E, alpha1E*J1E])
    magsys.add_coupling((0,1,0), 0, 1, Jdiag=[J1E, J1E, alpha1E*J1E])
    
    J1O  = - params['J1O']
    alpha1O  = params['alpha1O']
    magsys.add_coupling((0,0,0), 2, 3, Jdiag=[J1O, J1O, alpha1O*J1O])
    magsys.add_coupling((1,0,0), 2, 3, Jdiag=[J1O, J1O, alpha1O*J1O])
    magsys.add_coupling((0,1,0), 2, 3, Jdiag=[J1O, J1O, alpha1O*J1O])
    
    J2ab = params['J2ab']
    alpha2ab = params['alpha2ab']
    magsys.add_coupling((1,0,0), 2, 1, Jdiag=[J2ab, J2ab, alpha2ab*J2ab])
    magsys.add_coupling((0,1,0), 2, 1, Jdiag=[J2ab, J2ab, alpha2ab*J2ab])
    magsys.add_coupling((1,1,0), 2, 1, Jdiag=[J2ab, J2ab, alpha2ab*J2ab])
    magsys.add_coupling((1,0,1), 0, 3, Jdiag=[J2ab, J2ab, alpha2ab*J2ab])
    magsys.add_coupling((0,1,1), 0, 3, Jdiag=[J2ab, J2ab, alpha2ab*J2ab])
    magsys.add_coupling((1,1,1), 0, 3, Jdiag=[J2ab, J2ab, alpha2ab*J2ab])
    
    J2aa = params['J2aa']
    alpha2aa = params['alpha2aa']
    magsys.add_coupling((0,0,0), 2, 0, Jdiag=[J2aa, J2aa, alpha2aa*J2aa])
    magsys.add_coupling((1,0,0), 2, 0, Jdiag=[J2aa, J2aa, alpha2aa*J2aa])
    magsys.add_coupling((0,1,0), 2, 0, Jdiag=[J2aa, J2aa, alpha2aa*J2aa])
    magsys.add_coupling((0,0,1), 0, 2, Jdiag=[J2aa, J2aa, alpha2aa*J2aa])
    magsys.add_coupling((1,0,1), 0, 2, Jdiag=[J2aa, J2aa, alpha2aa*J2aa])
    magsys.add_coupling((0,1,1), 0, 2, Jdiag=[J2aa, J2aa, alpha2aa*J2aa])
    
    J2bb = params['J2bb']
    alpha2bb = params['alpha2bb']
    magsys.add_coupling((0,0,0), 3, 1, Jdiag=[J2bb, J2bb, alpha2bb*J2bb])
    magsys.add_coupling((1,0,0), 3, 1, Jdiag=[J2bb, J2bb, alpha2bb*J2bb])
    magsys.add_coupling((0,1,0), 3, 1, Jdiag=[J2bb, J2bb, alpha2bb*J2bb])
    magsys.add_coupling((0,0,1), 1, 3, Jdiag=[J2bb, J2bb, alpha2bb*J2bb])
    magsys.add_coupling((1,0,1), 1, 3, Jdiag=[J2bb, J2bb, alpha2bb*J2bb])
    magsys.add_coupling((0,1,1), 1, 3, Jdiag=[J2bb, J2bb, alpha2bb*J2bb])
    
    J3   = params['J3']
    alpha3   = params['alpha3']
    magsys.add_coupling((0,0,0), 3, 0, Jdiag=[J3, J3, alpha3*J3])
    magsys.add_coupling((0,0,1), 1, 2, Jdiag=[J3, J3, alpha3*J3])
    
    J9aa = params['J9aa']
    alpha9aa = params['alpha9aa']
    magsys.add_coupling((1,1,1), 2, 0, Jdiag=[J9aa, J9aa, alpha9aa*J9aa])
    magsys.add_coupling((1,1,2), 0, 2, Jdiag=[J9aa, J9aa, alpha9aa*J9aa])
    
    J9bb = params['J9bb']
    alpha9bb = params['alpha9bb']
    magsys.add_coupling((1,1,1), 3, 1, Jdiag=[J9bb, J9bb, alpha9bb*J9bb])
    magsys.add_coupling((1,1,2), 1, 3, Jdiag=[J9bb, J9bb, alpha9bb*J9bb])
    
    D = params['D']
    DM = [D, 0., 0.]
    # The "directions" for the DM interaction have the lattice's inversion and threefold rotation
    C3 = Rotation.from_rotvec(2.*np.pi/3. * np.array([0,0,1]))
    for sl in [0,2]:
        magsys.add_coupling((1,-1,0), sl, sl, D=DM)
        magsys.add_coupling((0, 1,0), sl, sl, D=C3.apply(DM, inverse=False))
        magsys.add_coupling((-1,0,0), sl, sl, D=C3.apply(DM, inverse=True))
    
    for sl in [1,3]:
        magsys.add_coupling((-1,1,0), sl, sl, D=DM)
        magsys.add_coupling((0,-1,0), sl, sl, D=C3.apply(DM, inverse=False))
        magsys.add_coupling((1, 0,0), sl, sl, D=C3.apply(DM, inverse=True))
    
    return magsys

def XXZ_3D_GS(params):
    ''' Compare ground state energy for different angles '''
    dim = 3
    
    thetas = np.linspace(0., np.pi/2., num=100)
    phis = np.linspace(0., np.pi, num=5)
    energies = np.zeros([len(thetas), len(phis)])
    
    spin_magnitudes = [1., 1., 1., 1.]
    
    for i_phi, phi in enumerate(phis):
        rot_axis = np.array([np.sin(phi), np.cos(phi), 0.])
        for i_theta, theta in enumerate(thetas):
            r0 = Rotation.from_rotvec( theta * rot_axis )
            r1 = Rotation.from_rotvec( (np.pi+theta) * rot_axis )
            sl_rotations = [r0, r0, r1, r1]
    
            magsys = bh.magnonsystem_t(dim, spin_magnitudes, sl_rotations)
    
            XXZ_3D_setinteractions(magsys, params)
    
            energies[i_theta,i_phi] = magsys.classical_energy()
    
    plt.plot(thetas, energies)
    plt.show()
    
    return

def XXZ_3D(k, params, r0, r1, verbose=False):

    dim = 3
    
    sl_rotations = [r0, r0, r1, r1]

    spin_magnitudes = [1., 1., 1., 1.]

    magsys = bh.magnonsystem_t(dim, spin_magnitudes, sl_rotations, verbose=True)
    
    XXZ_3D_setinteractions(magsys, params)
    
    if verbose:
        magsys.show()
    
    ####################################################################################
    lattice_vectors = [ [3./2., -np.sqrt(3.)/2., 0.], [3./2., np.sqrt(3.)/2., 0.], [-2., 0., 2.] ]
    
    ham, tau3 = magsys.bloch_ham(k, mode='cartesian', lattice_vectors=lattice_vectors)
    print(f'{magsys.spin_magnitudes = }')
        
    return ham, tau3


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.prog = 'XXZ_system.py'
    parser.epilog = 'Example usage: python3 XXZ_system.py'
    args = parser.parse_args()
    np.set_printoptions(linewidth=250)
    
    params = {
        'J1E': 1., 'alpha1E': 0.4, 
        'J1O': 1., 'alpha1O': 0.4, 
        'J2ab': 0.45, 'alpha2ab': 0.45, 
        'J2aa': 0.55, 'alpha2aa': 0.55, 
        'J2bb': 0.65, 'alpha2bb': 0.65, 
        'J3': 0.2, 'alpha3': 0.3, 
        'J9aa': 0.15, 'alpha9aa': 1., 
        'J9bb': 0.25, 'alpha9bb': 1., 
        'D': 0.35, 
        'H_Ea': 0.11, 
        'H_Eb': 0.12, 
        'H_Oa': 0.13, 
        'H_Ob': 0.14,
    }
    
#     XXZ_3D_GS(params)

    r0 = Rotation.from_rotvec( np.pi/2. * np.array([0,1,0]))
    r1 = Rotation.from_rotvec(-np.pi/2. * np.array([0,1,0]))
    
    print('\n*** Energy at a single momentum ***')
    k = [0.1, 0.2, 0.3]
    
    ham, tau3 = XXZ_3D(k, params, r0, r1)
    
    print(f'{k = }')
    print(f'{tau3.shape = }')
    print(f'{ham.shape = }')
    print(f'{(tau3 @ ham).shape = }')
    energy = np.sort( np.linalg.eigvals(tau3 @ ham) )
    print(f'{energy = }')
    
