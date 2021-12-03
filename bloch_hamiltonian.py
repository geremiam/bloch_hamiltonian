import argparse
import numpy as np
from numpy import pi, sqrt, cos, sin, exp, conj, real, imag
import scipy
from scipy.spatial.transform import Rotation
import math
import numbers
import copy

def ham_momentum_RL(q, couplings_dic, verbose=False):
    '''
    Computes the Bloch Hamiltonian at q (expressed in the basis of reciprocal lattice 
    vectors) based on coupling matrices for the different primitive translations.
    
    q : momentum expressed in the basis of primitive lattice vectors {b0, b1, ...}, i.e.
        k = q[0,...] b0 + q[1,...] b1 + ...
        The first dimension is the momentum components.
        The output of np.meshgrid(..., indexing='ij') can be used for q.
    
    couplings_dic : dictionary whose keys are the primitive translations (in the form of
                    tuples) and whose values are numpy arrays givin the coupling matrices
                    Ex.:
                    (n0,n1,...):np.array(square matrix), where R = n0 a0 + n1 a1 + ...
    
    return : Bloch Hamiltonian of the form sum_R e^{-i k . R} couplings_dic[R]
             The dot product gives 2 pi (n0 q[0,...] + n1 q[1,...] + ...)
    '''
    q = np.atleast_1d(q) # Make into numpy array
    
    dim = q.shape[0]
    q_shape = q.shape[1:]
    
    # Use an alias
    H = couplings_dic
    
    # Check that H[R] and H[-R] are Hermitian conjugates
    for R in H:
        R_neg = tuple(-np.array(R))
        assert np.allclose( H[R].T.conj(), H[R_neg] ), 'Oops! Looks like H[{}] and H[{}] are not Hermitian conjugates, though they should be.'.format(R,R_neg)
    
    # Get the shape of the coupling matrices
    keys_list = list( H.keys() )
    H_shape = H[keys_list[0]].shape
    
    if verbose:
        print(f'{H_shape = }')
        print(f'{keys_list = }')
    
    for key in keys_list:
        assert len(key)==dim, 'Length of keys in couplings_dic must match dimensionality. dim = {}, len({}) = {}'.format(dim, key, len(key))
        assert H[key].shape==H_shape, 'Coupling matrices in couplings_dic must have the same shapes'
    
    retval = np.zeros(q_shape+H_shape, complex)
    for R in H:
        # Calculate dot product and phase factor
        q_dot_R = np.tensordot( np.array(R, float), q, axes=[[0],[0]] )
        phase = np.exp(-2.j * np.pi * q_dot_R)
        phase = phase[..., None, None] # Add two length-one trailing dimensions
        # Add term
        retval += H[R] * phase
    
    assert np.allclose(np.swapaxes(retval.conj(),-1,-2), retval), 'Bloch Hamiltonian should have been Hermitian, but is not'
    
    return retval

def ham_momentum_cart(k, lattice_vectors, couplings_dic, verbose=False):
    ''' 
    Returns momentum-space hamiltonian based on coupling matrices between nearby sites.
    Broadcasting happens with momentum k.
    
    k : numpy array of shape (..., dim), where dim is the number of dimensions
    
    lattice_vectors : n-component list of n-component lists of floats
    
    couplings_dic: {(i1, etc. , idim):arr}, where arr is an (n,n) array
    
    return: array of shape (..., n, n)
    
    '''
    
    k = np.atleast_1d(k) # Converts scalars to 1D arrays. Ensures k is a numpy array
    
    # Determining spatial dimension from lattice_vectors
    dim = len(lattice_vectors)
    for el in lattice_vectors:
        assert len(el)==dim, 'Incorrect dimensionality of lattice vectors'
    
    if verbose: smartprint('dim', dim)
    if verbose: smartprint('lattice_vectors', lattice_vectors)
        
    # Numerous checks to be done ########################################################
    
    if dim!=1:
        assert k.shape[-1]==dim, 'Last dimension of k must be spatial dimensions'
    
    for key in couplings_dic.keys():
        assert len(key)==dim, 'Incorrect dimensionality of unit-cell indices in couplings_dic'
        
        key_neg = tuple(-np.array(key))
        assert np.allclose( couplings_dic[key].T.conj(), couplings_dic[key_neg] ), 'Oops! Looks like couplings_dic[{}] and couplings_dic[{}] are not Hermitian conjugates, though they should be.'.format(key,key_neg)
    
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
    """
    Class for carrying out linear spin wave calculations in translationally invariant
    systems. Also computes classical ground-state energy.
    
    Lattice dimensionality, spin magnitudes, and spin directions are specified at 
    construction.
    
    Couplings between spins are specified using either method add_coupling() or 
    method add_coupling_(). (For now, limited to bilinear spin interactions.)
    
    Zeeman field at each spin can be specified using method add_field().
    
    Classical ground-state energy is given by method classical_energy().
    
    Bloch coefficient matrix is given my method bloch_ham().
    
    Method show() prints many class attributes.
    """
    def __init__(self, dim, spin_magnitudes, sl_rotations):
        '''
        dim : int
            Dimensionality of the lattice.
        
        spin_magnitudes : list of (strictly positive) multiples of 1/2
            The magnitude of each spin within the magnetic unit cell
        
        sl_rotations : list of instances of scipy.spatial.transform.rotation.Rotation (each being a single rotation)
            Rotations setting the ordering direction of each spin in the magnetic unit 
            cell, starting from the z direction. For example, an identity leaves the spin 
            in the z direction, while a rotation by pi about the y direction brings the 
            spin to the -z direction.
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html#scipy.spatial.transform.Rotation
        '''
        # Check on argument 'dim'
        assert isinstance(dim, int), '"dim" must be an int'
        # Checks on argument 'spin_magnitudes'
        assert len(spin_magnitudes)==len(sl_rotations), 'Arguments "spin_magnitudes" and "sl_rotations" must have the same number of elements'
        for el in spin_magnitudes:
            assert isinstance(el, numbers.Real), 'Elements of "spin_magnitudes" must be real valued'
            assert math.isclose(0., el % 0.5), 'Elements of "spin_magnitudes" must be multiples of 0.5'
            assert el > 0., 'Elements of "spin_magnitudes" must be non-negative'
        # Checks on argument 'sl_rotations'
        for el in sl_rotations:
            assert isinstance(el, scipy.spatial.transform.rotation.Rotation), '"sl_rotations" must be a list of instances of scipy.spatial.transform.rotation.Rotation'
            assert el.single==True, 'Each element of "sl_rotations" must be a single rotation'
        
        #################################################################################
        
        self._N = np.array([[  1./sqrt(2.),  1./sqrt(2.), 0.],
                            [-1.j/sqrt(2.), 1.j/sqrt(2.), 0.],
                            [           0.,           0., 1.]])
        
        self.dim = dim # Spatial dimensionality of system
        
        self.spin_magnitudes = spin_magnitudes
        
        self.n_sl = len(sl_rotations) # Number of sublattices in the magnetic unit cell
        
        self.tau3 = np.diag([1.,-1.]*self.n_sl)
        
        self.sl_rotations = sl_rotations # List of rotation objects
        
        self.fields = {}
        self.fields_rot = {}
        
        self.couplings = {} # Dictionary in which to store couplings
        self.couplings_sym = {} # Dictionary in which to store couplings in a symmetric way
        self.couplings_sym_rot = {} # Dictionary in which to store the couplings in the local basis
        self.cal_M = {} # Dictionary in which to store the couplings in the "ladder basis"
        self.m = {} # Dictionary in which to store subblocks of the above
        
        return
    
    def check_for_coupling(self, tup):
        '''
        Returns true if coupling "tup" or the reversed coupling has already been defined.
        '''
        
        tup_rev = (tuple(-np.array(tup[0])), tup[2], tup[1]) # Coupling in reverse direction
        
        # Check for both directions
        tup_check     = tup     in self.couplings
        tup_rev_check = tup_rev in self.couplings
        
        # Redundant check
        assert not (tup_check and tup_rev_check), 'Intra-spin terms not expected'
        
        # Return true if either coupling is already present
        return tup_check or tup_rev_check
    
    def add_coupling_(self, R, sl1, sl2, Jtensor):
        '''
        Define tensor J in the interaction S_(R, sl1)^transpose J S_(0, sl2).
        
        R : tuple of ints of length "dim"
            The magnetic unit cell of the first spin, given in terms of the primitive 
            lattice vectors.
        
        sl1: int in range(n_sl)
            The sublattice index of the first spin
        
        sl2: int in range(n_sl)
            The sublattice index of the second spin
        
        Jtensor: array of shape (3,3)
            The interaction array between spins as shown above.
        '''
        # Checks on argument R
        assert isinstance(R, tuple), '"R" must be a tuple'
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
        
        # Assign value as given by user
        self.couplings[tup] = Jtensor
        
        # Symmetrize couplings, such that Jtensor_ji = Jtensor_ij^T
        tup_rev = (tuple(-np.array(R)), sl2, sl1)
        self.couplings_sym[tup]     = Jtensor   / 2.
        self.couplings_sym[tup_rev] = Jtensor.T / 2.
        
        for t in [tup, tup_rev]:
            # Compute the couplings as seen in the local bases
            self.couplings_sym_rot[t] = self.sl_rotations[t[1]].as_matrix().T @ self.couplings_sym[t] @ self.sl_rotations[t[2]].as_matrix()
            # Compute the couplings in the "ladder basis"
            self.cal_M[t] = self._N.T.conj() @ self.couplings_sym_rot[t] @ self._N
            # Sub-block of the above
            self.m[t] = self.cal_M[t][0:2,0:2]
        
        return
    
    def add_coupling(self, R, sl1, sl2, heisen=None, Jdiag=None, D=None, Gamma=None):
        '''
        Define tensor J in the interaction S_(R, sl1)^transpose J S_(0, sl2) using the 
        conventional interaction terms.
        
        R : tuple of ints of length "dim"
            The magnetic unit cell of the first spin, given in terms of the primitive 
            lattice vectors.
        
        sl1: int in range(n_sl)
            The sublattice index of the first spin
        
        sl2: int in range(n_sl)
            The sublattice index of the second spin
        
        heisen: float
            Heisenberg interaction between the spins. Conflicts with "Jdiag".
        
        Jdiag: array like, size (3,)
            Diagonal elements of the tensor J. Conflicts with "heisen".
        
        D: array like, size (3,)
            DM interaction between the two spins. Note that the order of the spins as 
            shown above is important.
        
        Gamma: array like, size (3,)
            Gamma interaction between the two spins.
        
        lattice_vectors : dim-element list of dim-element lists
                          Specifies the primitive lattice vectors.
        '''
        assert (heisen is None) or (Jdiag is None), 'Cannot provide both "heisen" and "Jdiag"'
        
        Jtensor = np.zeros([3,3], float)
        
        if heisen is not None:
            assert isinstance(heisen, float), ''
            Jtensor += heisen * np.eye(3)
            
        if Jdiag is not None:
            Jdiag = np.atleast_1d(Jdiag)
            assert Jdiag.shape==(3,), '"Jdiag" should have dimension (3,)'
            assert np.isrealobj(Jdiag), '"Jdiag" must be real valued'
            Jtensor += np.diag(Jdiag)
        
        if D is not None:
            D = np.atleast_1d(D)
            assert D.shape==(3,), '"D" should have dimension (3,)'
            assert np.isrealobj(D), '"D" must be real valued'
            D0 = D[0]
            D1 = D[1]
            D2 = D[2]
            Jtensor += np.array([[ 0.,  D2, -D1],
                                 [-D2,  0.,  D0],
                                 [ D1, -D0,  0.]])
        
        if Gamma is not None:
            Gamma = np.atleast_1d(Gamma)
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
        '''
        Use to set Zeeman field at the various sites in the magnetic unit cell.
        
        sl_list : int or list of ints
            Sublattice index or indices at which to set the given field
        
        field : array like, size (3,)
            The cartesian components of the Zeeman field at the specified sublattice(s)
        
        lattice_vectors : dim-element list of dim-element lists
            Specifies the primitive lattice vectors.
        '''
        # Make "sl_list" a list if it isn't yet
        if not hasattr(sl_list, '__iter__'):
            sl_list = [sl_list]
        # Checks on "sl_list"
        for sl in sl_list:
            assert sl in range(self.n_sl), 'A value of "sl_list" is outside the specified range. Problematic value is {}'.format(sl)
            assert sl not in self.fields, 'Field for sl = {} has already been specified'.format(sl)
        
        field = np.atleast_1d(field) # Ensures field is a numpy array
        assert np.isrealobj(field), 'Zeeman field must be real valued.'
        
        for sl in sl_list:
            self.fields[sl] = field
            self.fields_rot[sl] = self.sl_rotations[sl].apply(field, inverse=True)
            assert np.allclose(self.sl_rotations[sl].as_matrix().T @ field, self.sl_rotations[sl].apply(field, inverse=True)), 'Oops! May have gotten rotation directions wrong'
        
        return
    
    def classical_energy(self):
        '''
        Computes the classical energy.
        
        return: The classical energy (a float).
        '''
        accumulator = 0.
        
        # Contribution from bilinear spin interactions
        for tup in self.couplings_sym_rot:
            sl1 = tup[1]
            sl2 = tup[2]
            Jtilde_zz = self.couplings_sym_rot[tup][2,2]
            accumulator += self.spin_magnitudes[sl1] * Jtilde_zz * self.spin_magnitudes[sl2]
        
        # Contribution from Zeeman fields
        for sl in self.fields_rot:
            Btilde_z = self.fields_rot[sl][2]
            accumulator += - self.spin_magnitudes[sl] * Btilde_z
        
        return accumulator
    
    def coupling_matrices(self, verbose=False):
        '''
        Computes nonzero coupling matrices between lattice cells.
        
        return: dictionary whose keys are the primitive translations (in the form of
                tuples) and whose values are numpy arrays givin the coupling matrices
                Ex.:
                (n0,n1,...):np.array(square matrix), where R = n0 a0 + n1 a1 + ... is the 
                associated primitive translation.
        '''
        # Define h, as defined in docs
        # Need to make deep copy in order to avoid modified self.m
        h = copy.deepcopy(self.m)
        
        # Adding diagonal contribution #################################################
        
        # Ensure h has diagonal terms for R=0
        # These do not exist at this point and so the check is redundant, but I'm still implementing it for future-proofing
        for sl in range(self.n_sl):
            tup_diag = ( tuple(np.zeros(self.dim,int)), sl, sl ) # Tuples indexing the diagonal components
            if tup_diag not in h: # Create a (zero-valued) dictionary entry if necessary
                h[tup_diag] = np.zeros([2,2], complex)
        
        for sl in self.fields_rot:
            tup_diag = ( tuple(np.zeros(self.dim,int)), sl, sl ) # Tuples indexing the diagonal components
            
            # Add Zeeman field contribution
            Btilde_z = self.fields_rot[sl][2]
            h[tup_diag] += Btilde_z * np.eye(2)
        
        for sl in range(self.n_sl):
            tup_diag = ( tuple(np.zeros(self.dim,int)), sl, sl ) # Tuples indexing the diagonal components
            # Add other contribution
            accumulator = 0.
            for tup in self.couplings_sym_rot:
                R = tup[0]
                sl1 = tup[1]
                sl2 = tup[2]
                
                if sl2==sl:
                    accumulator += - self.spin_magnitudes[sl1] * self.couplings_sym_rot[tup][2,2] / 2.
                if sl1==sl:
                    accumulator += - self.spin_magnitudes[sl2] * self.couplings_sym_rot[tup][2,2] / 2.
            h[tup_diag] += accumulator * np.eye(2)
        
        
        # Building H out of the blocks of h #############################################
        
        # Create an array H for each R
        H = {}
        # Creates all the needed keys, with zero-valued arrays as values
        for tup in h:
            R = tup[0]
            H[R] = np.zeros([2*self.n_sl, 2*self.n_sl], complex)
        
        # Populating the H arrays with the h arrays
        for tup in h:
            R = tup[0]
            sl1 = tup[1]
            sl2 = tup[2]
            
            # Block in which to place h[tup]
            inds = ( slice(2*sl1, 2*sl1+2), slice(2*sl2, 2*sl2+2) )
            
            H[R][inds] += h[tup]
        
        if verbose:
            print('h =')
            for key, val in h.items():
                print('\n{} -> \n{}'.format(key, val))
            print('*'*80)
        
            print('H =')
            for key, val in H.items():
                print('\n{} -> \n{}'.format(key, val))
            print('*'*80)
        
        # Check that H[R] and H[-R] are Hermitian conjugates
        for R in H:
            R_neg = tuple(-np.array(R))
            assert np.allclose( H[R].T.conj(), H[R_neg] ), 'Oops! Looks like H[{}] and H[{}] are not Hermitian conjugates, though they should be.'.format(R,R_neg)
        
        return H
    
    def bloch_ham(self, k, mode, lattice_vectors=None):
        '''
        Computes Bloch coefficient matrix.
        
        k : array, (dim, ...)
            Numpy array of momenta. First dimension is the momentum component.
            The output of np.meshgrid(.., indexing='ij') can be directly used for k.
            For 1D systems, the length-one "dim" axis can be omitted.
        
        mode : either "RL" or "cartesian"
            Specifies the coordinates used to express momentum. If "RL", momentum is 
            expressed in terms of the reciprocal lattice vectors. If "cartesian", the 
            momentum is expressed in Cartesian coordinates, and the argument 
            "lattice_vectors" is needed.
        
        lattice_vectors : dim-element list of dim-element lists
            Specifies the primitive lattice vectors.
        
        returns:
            ham : shape (..., 2*n_sl, 2*n_sl)
                Bloch coefficient matrix for the given momenta.
            
            tau3 : shape (2*n_sl, 2*n_sl)
                Para-unitary identity corresponding to the given coefficient matrix.
                The energies are given by the positive eigenvalues of tau3 @ ham.
        '''
        
        # Checks on argument "k"
        if self.dim==1:
            k = np.atleast_2d(k)
        else:
            k = np.atleast_1d(k)
            if k.ndim==1:
                k = k[..., None]
        
        assert k.shape[0]==self.dim, 'First dimension of k must match the dimensionality'
        
        # Identify mode
        modes = ['RL', 'cartesian']
        assert mode in modes, 'mode "{}" is not valid'.format(mode)
        
        if mode==modes[0]: # Mode 'RL'
            q = k
        elif mode==modes[1]: # Mode 'cartesian'
            assert lattice_vectors is not None, 'In mode "cartesian", argument lattice_vectors is needed'
            assert len(lattice_vectors)==self.dim, 'Length of lattice_vectors must match spatial dimension'
            for el in lattice_vectors:
                assert len(el)==self.dim, 'Length of individual components of lattice_vectors must match spatial dimension'
                for i in el:
                    assert isinstance(i, numbers.Real), 'Components of lattice vectors must be real valued'
            
            lattice_vectors = np.atleast_2d(lattice_vectors) # Turn into numpy array
            # Use the lattice vectors to do the basis transformation to the RL basis
            q = (1./(2.*np.pi)) * np.tensordot( lattice_vectors, k, axes=1 )
        
        H = self.coupling_matrices() # Get coupling matrices
        
        ham = ham_momentum_RL(q, H) # Get the Bloch coefficient matrix
        
        ham = np.squeeze(ham) # Gets rid of length-one dimensions
        
        return ham, self.tau3
    
    def show(self):
        '''
        Prints many class attributes to help with debugging.
        '''
        print(f'{self.dim = }')
        print(f'{self.spin_magnitudes = }')
        print(f'{self.fields = }')
        
        print('\nself.couplings =')
        for key, val in self.couplings.items():
            print('\n{} -> \n{}'.format(key, val))
        print('*'*80)
    
        print('self.couplings_sym =')
        for key, val in self.couplings_sym.items():
            print('\n{} -> \n{}'.format(key, val))
        print('*'*80)
    
        print('self.cal_M =')
        for key, val in self.cal_M.items():
            print('\n{} -> \n{}'.format(key, val))
        print('*'*80)
        
        print('self.m =')
        for key, val in self.m.items():
            print('\n{} -> \n{}'.format(key, val))
        print('*'*80)
        
        self.coupling_matrices(verbose=True)
        
        print(f'{self.classical_energy() = }')
        
        return

def test():
    np.set_printoptions(linewidth=250)
    
    dim = 2
    
    r0 = Rotation.identity()
    r1 = Rotation.from_rotvec(pi * np.array([0,1,0]))
    sl_rotations = [r0, r1]
    
    spin_magnitudes = [0.5, 0.5]
    
    magnonsystem = magnonsystem_t(dim, spin_magnitudes, sl_rotations)
    
    
#     magnonsystem.add_field(0, [0,0,0.1])
#     magnonsystem.add_field(1, [0,0,-0.3])
    
#     magnonsystem.add_coupling_((1,0), 1, 1, np.ones([3,3]))
#     magnonsystem.add_coupling_((1,0), 0, 1, np.ones([3,3]))
#     magnonsystem.add_coupling_((1,0), 1, 0, np.ones([3,3]))
#     
#     magnonsystem.add_coupling_((0,1), 0, 0, np.ones([3,3]))
#     magnonsystem.add_coupling_((0,1), 1, 1, np.ones([3,3]))
#     magnonsystem.add_coupling_((0,1), 0, 1, np.ones([3,3]))
    
    magnonsystem.add_coupling((0,0), 0, 1, heisen=1., D=[0, 0, 0])
    
    magnonsystem.add_coupling((1,0), 0, 0, Jdiag=[1,1,1])
    
    magnonsystem.show()
    
    k1 = np.linspace(-np.pi, np.pi, num=10)
    k2 = np.linspace(-np.pi, np.pi, num=15)
    k = np.meshgrid(k1 ,k2, indexing='ij')
    
    blochham, tau3 = magnonsystem.bloch_ham(k, 'RL')
    print(f'{blochham.shape = }')
    
    lattice_vectors = [[1,0],[0,1]]
    blochham, tau3 = magnonsystem.bloch_ham(k, 'cartesian', lattice_vectors=lattice_vectors)
    print(f'{blochham.shape = }')
    
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.prog = "bloch_hamiltonian.py"
    parser.description = "Defines object for calculating Bloch Hamiltonians for various systems."
    args = parser.parse_args()
    
    test()
