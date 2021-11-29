import argparse
import numpy as np
from numpy import pi, sqrt, cos, sin, exp, conj, real, imag
import scipy
from scipy.spatial.transform import Rotation
import math
import numbers

# k is an array whose last dimension gives Cartesian components of momentum (of length 3)

def ham_momentum(k, lattice_vectors, couplings_dic, verbose=False):
    ''' 
    Returns momentum-space hamiltonian based on coupling matrices between nearby sites.
    Broadcasting happens with momentum k.
    
    k : numpy array of shape (..., dim), where dim is the number of dimensions
    lattice_vectors : n-component list of n-component lists of floats
    couplings_dic: {(i1, etc. , idim):arr}, where arr is an (n,n) array
    
    return: array of shape (..., n, n)
    
    '''
    
    k = np.atleast_1d(k) # Converts scalars to 1D arrays. Ensures k is a numpy array
    
    # Determining spatial dimension from lattice_vectors ################################
    # If dimension is 1, lattice_vectors does not need to be a list of lists.
    
    if isinstance(lattice_vectors, list):
        dim = len(lattice_vectors)
        
        if dim==1: # Make it a list of lists if it isn't yet
            if not isinstance(lattice_vectors[0], list):
                lattice_vectors = [lattice_vectors]
    else: # If lattice_vectors is just a number, take spatial dimension to be 1.
        dim = 1
        lattice_vectors = [[lattice_vectors]] # Turn into list of lists
    
    if verbose: smartprint('dim', dim)
    if verbose: smartprint('lattice_vectors', lattice_vectors)
        
    # Numerous checks to be done ########################################################
    
    for el in lattice_vectors:
        assert len(el)==dim, 'Incorrect dimensionality of lattice vectors'
    
    # Ensure the keys of the dictionary are all tuples
    couplings_dic = make_keys_tuples(couplings_dic)
    
    if dim!=1:
        assert k.shape[-1]==dim, 'Last dimension of k must be spatial dimensions'
    
    for key in couplings_dic.keys():
        assert len(key)==dim, 'Incorrect dimensionality of unit-cell indices in couplings_dic'
        
        for i in key:
            assert isinstance(i, int), 'Indices appearing in couplings_dic must be integers'
        
        if not np.any(np.array(key)): # If the indices are all zero
            assert np.allclose( couplings_dic[key], couplings_dic[key].T.conj() ), 'Zero-momentum term must be Hermitian'
        else: # If the indices are not all zero
            assert tuple(-np.array(key)) not in couplings_dic, 'couplings_dic should not contain both a vector and its additive inverse'
    
    values_list = list(couplings_dic.values())
    for i in range(len(values_list)):
        assert values_list[i].shape==values_list[0].shape, 'Matrices in "couplings_dic" should all have the same shape'
        assert len(values_list[i].shape)==2, 'Matrices in "couplings_dic" should be two-dimensional'
        matrix_shape = values_list[i].shape
    
    # Actual calculation ###############################################################
    
    if dim==1: # Add a last dimension to k if dim==1
        k = k[...,None]
    if verbose: smartprint('k.shape',k.shape)
    
    output = np.zeros(k.shape[:-1] + matrix_shape, complex) # Shape of output matrix
    if verbose: smartprint('output.shape',output.shape)
    
    for key in couplings_dic:
        R = np.dot( np.array(lattice_vectors), np.atleast_1d(np.array(key)) )
        if verbose: smartprint('R',R)
        
        dotprod = np.dot(k, R)
        if verbose: smartprint('dotprod.shape', dotprod.shape)
        
        contribution = np.exp(1.j*dotprod)[...,None,None] * couplings_dic[key]
        
        if not np.any(np.array(key)): # If the indices are all zero
            output += contribution
        else:
            output += contribution + np.swapaxes(contribution, -1, -2).conj()
    
    output = np.squeeze(output) # Gets rid of length-one dimensions (specifically, the momentum)
    
    return output

class magnonsystem_t:
    """ Class description. 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html#scipy.spatial.transform.Rotation
    """
    def __init__(self, dim, spin_magnitudes, sl_rotations):
        # Check on argument 'dim'
        assert isinstance(dim, int), '"dim" must be an int'
        # Checks on argument 'spin_magnitudes'
        assert len(spin_magnitudes)==len(sl_rotations), 'Arguments "spin_magnitudes" and "sl_rotations" must have the same number of elements'
        for el in spin_magnitudes:
            assert isinstance(el, numbers.Real), 'Elements of "spin_magnitudes" must be real valued'
            assert math.isclose(0., el % 0.5), 'Elements of "spin_magnitudes" must be non-negative multiples of 0.5'
        # Checks on argument 'sl_rotations'
        for el in sl_rotations:
            assert isinstance(el, scipy.spatial.transform.rotation.Rotation), '"sl_rotations" must be a list of instances of scipy.spatial.transform.rotation.Rotation'
            assert el.single==True, 'Each element of "sl_rotations" must be a single rotation'
        
        #################################################################################
        
        self.dim = dim # Spatial dimensionality of system
        
        self.spin_magnitudes = spin_magnitudes
        
        self.n_sl = len(sl_rotations) # Number of sublattices in the magnetic unit cell
        
        self.sl_rotations = sl_rotations # List of rotation objects
        
        self.couplings = {} # List in which to store couplings
        self.couplings_rot = {} # List in which to store the couplings in the local basis
        
        self.fields = {}
        self.fields_rot = {}
    
    def check_for_coupling(self, tup):
        
        tup_rev = (tuple(-np.array(tup[0])), tup[2], tup[1])
        
        tup_check     = tup     in self.couplings
        tup_rev_check = tup_rev in self.couplings
        
        assert not (tup_check and tup_rev_check), 'Intra-spin terms not expected'
        
        return tup_check or tup_rev_check
    
    def add_coupling_(self, R, sl1, sl2, Jtensor):
        # Checks on argument R
        assert isinstance(R, tuple), '"R" must be an tuple'
        assert len(R)==self.dim, '"R" should have dim = {} components'.format(self.dim)
        for i in range(self.dim):
            assert isinstance(R[i], int), '"R" must contain ints'
        # Checks on arguments sl1 and sl2
        assert isinstance(sl1, int), '"sl1" must be an int'
        assert isinstance(sl2, int), '"sl2" must be an int'
        assert sl1 in range(self.n_sl), '"sl1" exceeded given range'
        assert sl2 in range(self.n_sl), '"sl2" exceeded given range'
        # Checks on argument Jtensor
        assert isinstance(Jtensor, np.ndarray), '"Jtensor" must be an array'
        assert Jtensor.shape==(3,3), '"Jtensor" should have dimensions (3,3)'
        assert np.isrealobj(Jtensor), '"Jtensor" must be real valued'
        
        #################################################################################
        
        # Tuple to use as dictionary key
        tup = (R, sl1, sl2)
        
        # Refuse "self interaction" terms
        assert tup != (tuple(np.zeros([self.dim], int)), 0, 0), 'Intra-spin terms not handled'
        
        # Check that the term hasn't already been added
        assert not self.check_for_coupling(tup), 'This term or its inverse has already been added'
        
        # Assign value
        self.couplings[tup] = Jtensor
        self.couplings_rot[tup] = self.sl_rotations[sl1].as_matrix().T @ Jtensor @ self.sl_rotations[sl2].as_matrix().T
        
        return
    
    def add_coupling(self, R, sl1, sl2, heisen=None, Jdiag=None, D=None, Gamma=None):
        assert (heisen is None) or (Jdiag is None), 'Cannot provide both "heisen" and "Jdiag"'
        
        Jtensor = np.zeros([3,3], float)
        
        if heisen is not None:
            assert isinstance(heisen, float), ''
            Jtensor += heisen * np.eye(3)
            
        if Jdiag is not None:
            assert isinstance(Jdiag, np.ndarray), '"Jdiag" must be a vector'
            assert Jdiag.shape==(3,), '"Jdiag" should have dimension (3,)'
            assert np.isrealobj(Jdiag), '"Jdiag" must be real valued'
            Jtensor += np.diag(Jdiag)
        
        if D is not None:
            assert isinstance(D, np.ndarray), '"D" must be a vector'
            assert D.shape==(3,), '"D" should have dimension (3,)'
            assert np.isrealobj(D), '"D" must be real valued'
            D0 = D[0]
            D1 = D[1]
            D2 = D[2]
            Jtensor += np.array([[ 0.,  D2, -D1],
                                 [-D2,  0.,  D0],
                                 [ D1, -D0,  0.]])
        
        if Gamma is not None:
            assert isinstance(Gamma, np.ndarray), '"Gamma" must be a vector'
            assert Gamma.shape==(3,), '"Gamma" should have dimension (3,)'
            assert np.isrealobj(Gamma), '"Gamma" must be real valued'
            G0 = Gamma[0]
            G1 = Gamma[1]
            G2 = Gamma[2]
            Jtensor += np.array([[0., G2, G1],
                                 [G2, 0., G0],
                                 [G1, G0, 0.]])
        
        self.add_coupling_(R, sl1, sl2, Jtensor)
        
        return
    
    def add_field(self, sl_list, field):
        # Make "sl_list" a list if it isn't yet
        if not hasattr(sl_list, '__iter__'):
            sl_list = [sl_list]
        # Checks on "sl_list"
        for sl in sl_list:
            assert sl in range(self.dim), 'A value of "sl_list" is outside the specified range. Problematic value is {}'.format(sl)
            assert sl not in self.fields, 'Field for sl = {} has already been specified'.format(sl)
        
        field = np.atleast_1d(field) # Ensures field is a numpy array
        
        for sl in sl_list:
            self.fields[sl] = field
            self.fields_rot[sl] = self.sl_rotations[sl].as_matrix().T @ field
            assert np.allclose(self.sl_rotations[sl].as_matrix().T @ field, self.sl_rotations[sl].apply(field, inverse=True))
        
        return
    
    def classical_energy(self):
        accumulator = 0.
        
        for tup in self.couplings_rot:
            sl1 = tup[1]
            sl2 = tup[2]
            Jtilde_zz = self.couplings_rot[tup][2,2]
            accumulator += self.spin_magnitudes[sl1] * Jtilde_zz * self.spin_magnitudes[sl2]
        
        for sl in self.fields_rot:
            Btilde_z = self.fields_rot[sl][2]
            accumulator += - self.spin_magnitudes[sl] * Btilde_z
        
        return accumulator
    
    def bloch_ham(self, k):
        return

def test():
    dim = 2
    
    r0 = Rotation.identity()
    r1 = Rotation.from_rotvec(pi * np.array([0,1,0]))
    sl_rotations = [r0, r1]
    
    spin_magnitudes = [0.5, 0.5]
    
    magnonsystem = magnonsystem_t(dim, spin_magnitudes, sl_rotations)
    print(f'{magnonsystem.dim = }')
    
    magnonsystem.add_coupling_((0,0), 0, 1, np.ones([3,3]))
    
    magnonsystem.add_coupling_((1,0), 0, 0, np.ones([3,3]))
    magnonsystem.add_coupling_((1,0), 1, 1, np.ones([3,3]))
    magnonsystem.add_coupling_((1,0), 0, 1, np.ones([3,3]))
    magnonsystem.add_coupling_((1,0), 1, 0, np.ones([3,3]))
    
    magnonsystem.add_coupling_((0,1), 0, 0, np.ones([3,3]))
    magnonsystem.add_coupling_((0,1), 1, 1, np.ones([3,3]))
    magnonsystem.add_coupling_((0,1), 0, 1, np.ones([3,3]))
    
    magnonsystem.add_coupling((1,1), 0, 0, Jdiag=np.array([0,0,1]))
    
    magnonsystem.add_field(0, [0,0,0.1])
    magnonsystem.add_field(1, [0,0,-0.3])
    
    print(f'{magnonsystem.couplings = }')
    print(f'{magnonsystem.fields = }')
    print(f'{magnonsystem.classical_energy() = }')


if __name__ == "__main__":
    test()
