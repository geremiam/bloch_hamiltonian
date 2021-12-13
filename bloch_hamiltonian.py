import argparse
import numpy as np
from numpy import sqrt
import scipy
from scipy.spatial.transform import Rotation
import math
import numbers

def ham_momentum_RL_dic(q, couplings_dic, verbose=False, safety=True):
    '''
    Computes the Bloch Hamiltonian at q (expressed in the basis of reciprocal lattice 
    vectors) based on coupling matrices for the different primitive translations.
    
    q : momentum expressed in the basis of primitive lattice vectors {b0, b1, ...}, i.e.
        k = q[0,...] b0 + q[1,...] b1 + ...
        The first dimension is the momentum components; other dimensions are arbitrary.
        The output of np.meshgrid(..., indexing='ij') can be used for q.
    
    couplings_dic : dictionary whose keys are the primitive translations (in the form of
                    tuples) and whose values are numpy arrays giving the coupling matrices
                    Ex.:
                    (n0,n1,...):np.array(square matrix), where R = n0 a0 + n1 a1 + ...
    
    return : Bloch Hamiltonian of the form sum_R e^{-i k . R} couplings_dic[R]
             The dot product gives 2 pi (n0 q[0,...] + n1 q[1,...] + ...)
    '''
    q = np.atleast_1d(q) # Make into numpy array
    
    if safety: dim = q.shape[0]
    q_shape = q.shape[1:]
    
    # Use an alias
    H = couplings_dic
    
    # Check that H[R] and H[-R] are Hermitian conjugates
    if safety:
        for R in H:
            R_neg = tuple(-np.array(R))
            assert np.allclose( H[R].T.conj(), H[R_neg] ), 'Oops! Looks like H[{}] and H[{}] are not Hermitian conjugates, though they should be.'.format(R,R_neg)
    
    # Get the shape of the coupling matrices
    keys_list = list( H.keys() )
    H_shape = H[keys_list[0]].shape
    
    if verbose:
        print(f'{H_shape = }')
        print(f'{keys_list = }')
    
    if safety:
        for key in keys_list:
            assert len(key)==dim, 'Length of keys in couplings_dic and dimensionality must match. dim = {}, len({}) = {}'.format(dim, key, len(key))
            assert H[key].shape==H_shape, 'Coupling matrices in couplings_dic must have the same shapes'
            assert H[key].ndim==2, 'Coupling matrices couplings_dic must be two dimensional'
    
    retval = np.zeros(q_shape+H_shape, complex)
    for R in H:
        # Calculate dot product and phase factor
        q_dot_R = np.tensordot( np.array(R, float), q, axes=[[0],[0]] )
        phase = np.exp(-2.j * np.pi * q_dot_R)
        phase = phase[..., None, None] # Add two length-one trailing dimensions
        # Add term
        retval += H[R] * phase
    
    if safety: assert np.allclose(np.swapaxes(retval.conj(),-1,-2), retval), 'Bloch Hamiltonian should have been Hermitian, but is not'
    
    return retval

def ham_momentum_RL(q, R_array, coupling_mat_array, verbose=False, safety=True):
    '''
    Computes the Bloch Hamiltonian at q (expressed in the basis of reciprocal lattice 
    vectors) based on coupling matrices for the different primitive translations.
    
    q : Numpy array of shape (n_dim, ...)
        Momentum expressed in the basis of primitive lattice vectors {b0, b1, ...}, i.e.
        k = q[0,...] b0 + q[1,...] b1 + ...
        The first dimension is the momentum components; other dimensions are arbitrary.
        The output of np.meshgrid(..., indexing='ij') can be used for q.
    
    R_array : numpy array of ints of shape (n_ints, dim)
        Two-dimensional numpy array whose first dimension indexes the different lattice 
        translations R with nonzero coupling matrices, and whose second dimension are the 
        (integer-valued) components of the lattice translations, given as
        R = n0 a0 + n1 a1 + ...
        Order of the first dimension must correspond to that of coupling_mat_array.
    
    coupling_mat_array : numpy array of floats of shape (n_ints, n_orb, n_orb)
        dimensional numpy array whose first dimension indexes the different lattice 
        translations R with nonzero coupling matrices, and whose second and third 
        dimensions are the matrix dimensions for the coupling matrices.
        Order of the first dimension must correspond to that of R_array.
    
    safety: Boolean, default True
        If False, certain checks, like shape of input arrays and Hermicity of output, are 
        not done. Cuts down on calculation time if "q" contains few momenta.
    
    return : numpy array of shape (..., n_orb, n_orb)
        Bloch Hamiltonian of the form sum_R e^{-i k . R} couplings_dic[R]
        The dot product gives 2 pi (n0 q[0,...] + n1 q[1,...] + ...)
    '''
    # q.shape = (n_dim, ...)
    # R_array.shape = (n_ints, n_dim)
    # coupling_mat_array.shape = (n_ints, n_orb, n_orb)
    
    if safety:
        assert R_array.ndim==2
        assert coupling_mat_array.ndim==3
        assert coupling_mat_array.shape[1]==coupling_mat_array.shape[2]
    
    # q_dot_R.shape = (..., n_ints)
    q_dot_R = np.tensordot(q, R_array, axes=[[0],[-1]])
    # phase.shape = (..., n_ints)
    phase = np.exp(-2.j * np.pi * q_dot_R)
    
    # retval.shape = (..., n_orb, n_orb)
    retval = np.tensordot(phase, coupling_mat_array, axes=1)
    
    if safety:
        assert np.allclose(np.swapaxes(retval.conj(),-1,-2), retval), 'Bloch Hamiltonian should have been Hermitian, but is not'
    
    if verbose:
        print(f'{q.shape = }')
        print(f'{R_array.shape = }')
        print(f'{coupling_mat_array.shape = }')
        print(f'{phase.shape = }')
        print(f'{retval.shape = }')
    
    return retval


class magnonsystem_t:
    """
    Class for carrying out linear spin wave calculations in translationally invariant
    systems. Also computes classical ground-state energy.
    
    Lattice dimensionality, spin magnitudes, and spin directions are specified at 
    construction.
    
    Couplings between spins are specified using either method add_coupling() or 
    method add_coupling_(). (For now, limited to bilinear spin interactions.)
    Must be called before bloch_ham() is first called.
    
    Zeeman field at each spin can be specified using method add_field().
    Must be called before bloch_ham() is first called.
    
    Classical ground-state energy (order S^2 terms) is given by method classical_energy().
    
    Bloch coefficient matrix (order S^1 terms) is given my method bloch_ham().
    If calculating for a single momentum (or few momenta), significant speedup can be 
    achieved by setting safety=False.
    
    Method show() prints many class attributes.
    """
    def __init__(self, dim, spin_magnitudes, sl_rotations, verbose=False):
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
        
        verbose: boolean (default False)
            If True, prints the magnitude and classical direction of each spin in the 
            magnetic unit cell.
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
        
        self.H = {}
        self.H_keys = None
        self.H_vals = None
        
        if verbose:
            print('Spin magnitudes and directions:')
            for sl in range(self.n_sl):
                print()
                print('\tSublattice {}'.format(sl))
                print('\tspin magnitude: {}'.format(self.spin_magnitudes[sl]))
                spin_dir = self.spin_dir(sl)
                print('\tspin direction: {}'.format(spin_dir))
            print('*'*80)
        
        return
    
    def spin_dir(self, sl):
        '''
        Returns the classical spin direction for the given sublattice.
        
        sl: element of range(n_sl)
            A sublattice index
        
        return: array of shape (3,)
            Unit vector giving the spin direction on sublattice "sl"
        '''
        return self.sl_rotations[sl].apply([0., 0., 1.])
    
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
        Define tensor J in the interaction S_(R, sl1)^transpose J S_(0, sl2). Will refuse 
        if a coupling has already been defined for the given bond (in either direction).
        
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
        assert self.H=={}, 'Must define all couplings/interactions before bloch_ham() is first called'
        
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
        assert tup != ((0,)*self.dim, 0, 0), 'Intra-spin terms not handled'
        
        # Check that the term hasn't already been added
        assert not self.check_for_coupling(tup), 'This term or its inverse has already been added'
        
        # Assign value as given by user
        self.couplings[tup] = Jtensor
        
        # Symmetrize couplings, such that Jtensor_ji = Jtensor_ij^T
        tup_rev = (tuple(-np.array(R)), sl2, sl1)
        self.couplings_sym[tup]     = Jtensor
        self.couplings_sym[tup_rev] = Jtensor.T
        
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
        conventional interaction terms. Will refuse if a coupling has already been 
        defined for the given bond (in either direction).
        
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
            shown above is important. Gives the following: tensor components :
            Jtensor[beta,gamma] = epsilon[alpha,beta,gamma] D[alpha].
        
        Gamma: array like, size (3,)
            Gamma interaction between the two spins. Gives the following: tensor 
            components :
            Jtensor[beta,gamma] = |epsilon[alpha,beta,gamma]| Gamma[alpha].
        
        lattice_vectors : dim-element list of dim-element lists
                          Specifies the primitive lattice vectors.
        '''
        assert (heisen is None) or (Jdiag is None), 'Cannot provide both "heisen" and "Jdiag"'
        
        Jtensor = np.zeros([3,3], float)
        
        if heisen is not None:
            assert isinstance(heisen, numbers.Real), '"heisen" should be a real number type'
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
        assert self.H=={}, 'Must define all Zeeman fields before bloch_ham() is first called'
        
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
        Computes the classical energy per magnetic unit cell (order S^2 terms).
        
        return: The classical energy per magnetic unit cell (a float).
        '''
        accumulator = 0.
        
        # Contribution from bilinear spin interactions
        for tup in self.couplings_sym_rot:
            sl1 = tup[1]
            sl2 = tup[2]
            Jtilde_zz = self.couplings_sym_rot[tup][2,2]
            accumulator += 0.5 * self.spin_magnitudes[sl1] * Jtilde_zz * self.spin_magnitudes[sl2]
        
        # Contribution from Zeeman fields
        for sl in self.fields_rot:
            Btilde_z = self.fields_rot[sl][2]
            accumulator += - self.spin_magnitudes[sl] * Btilde_z
        
        return accumulator
    
    def coupling_matrices(self, verbose=False):
        '''
        Computes nonzero coupling matrices between lattice cells (order S^1 terms).
        
        Saved to self.H:
            dictionary whose keys are the primitive translations (in the form of tuples) 
            and whose values are numpy arrays givin the coupling matrices
            Ex.:
            (n0,n1,...):np.array(square matrix), where R = n0 a0 + n1 a1 + ... is the 
            associated primitive translation.
        
        Saved to self.H_keys:
            Keys of the dictionary self.H
        
        Saved to self.H_vals:
            Values of the dictionary self.H
        '''
        # Define h, as defined in docs
        h = {}
        # One contribution to h comes from m, up to a factor involving spin magnitudes
        for key in self.m:
            sl1 = key[1]
            sl2 = key[2]
            h[key] = np.sqrt(self.spin_magnitudes[sl1] * self.spin_magnitudes[sl2]) * self.m[key]
        
        # Adding diagonal contribution #################################################
        
        for sl in range(self.n_sl): # Cycle through sublattices
            tup_diag = ( (0,)*self.dim, sl, sl ) # Tuples indexing the diagonal components
            
            # Ensure h has diagonal terms for R=0
            # These do not exist at this point and so the check is redundant, but I'm still implementing it for future-proofing
            if tup_diag not in h: # Create a (zero-valued) dictionary entry if necessary
                h[tup_diag] = np.zeros([2,2], complex)
            
            # Add Zeeman field contribution
            if sl in self.fields_rot:
                Btilde_z = self.fields_rot[sl][2]
                h[tup_diag] += Btilde_z * np.eye(2)
            
            # Add other contribution (from the zz components of couplings_sym_rot)
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
        # Reset self.H
        self.H = {}
        # Creates all the needed keys, with zero-valued arrays as values
        for tup in h:
            R = tup[0]
            self.H[R] = np.zeros([2*self.n_sl, 2*self.n_sl], complex)
        
        # Populating the H arrays with the h arrays
        for tup in h:
            R = tup[0]
            sl1 = tup[1]
            sl2 = tup[2]
            
            # Block in which to place h[tup]
            inds = ( slice(2*sl1, 2*sl1+2), slice(2*sl2, 2*sl2+2) )
            
            self.H[R][inds] += h[tup]
        
        if verbose:
            print('h =')
            for key, val in h.items():
                print('\n{} -> \n{}'.format(key, val))
            print('*'*80)
        
        # Check that self.H[R] and self.H[-R] are Hermitian conjugates
        for R in self.H:
            R_neg = tuple(-np.array(R))
            assert np.allclose( self.H[R].T.conj(), self.H[R_neg] ), 'Oops! Looks like self.H[{}] and self.H[{}] are not Hermitian conjugates, though they should be.'.format(R,R_neg)
        
        # Extracting keys and values back to back is guaranteed to give the same order
        self.H_keys = np.array(list(self.H.keys()))
        self.H_vals = np.array(list(self.H.values()))
        
        return
    
    def bloch_ham(self, k, mode, lattice_vectors=None, squeeze_output=True, safety=True):
        '''
        Computes Bloch coefficient matrix (order S^1 terms).
        
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
        
        squeeze_output: Boolean, default True
            If True, length-one dimensions of the output array are suppressed.
        
        safety: Boolean, default True
            If False, certain checks, like shape of input arrays and Hermicity of output, 
            are not done. Cuts down on calculation time if "q" contains few momenta.
        
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
        
        if safety: assert k.shape[0]==self.dim, 'First dimension of k must match the dimensionality'
        
        # Identify mode
        modes = ['RL', 'cartesian']
        if safety: assert mode in modes, 'mode "{}" is not valid'.format(mode)
        
        if mode==modes[0]: # Mode 'RL'
            q = k
        elif mode==modes[1]: # Mode 'cartesian'
            if safety:
                assert lattice_vectors is not None, 'In mode "cartesian", argument lattice_vectors is needed'
                assert len(lattice_vectors)==self.dim, 'Length of lattice_vectors must match spatial dimension'
                for el in lattice_vectors:
                    assert len(el)==self.dim, 'Length of individual components of lattice_vectors must match spatial dimension'
                    for i in el:
                        assert isinstance(i, numbers.Real), 'Components of lattice vectors must be real valued'
            
            lattice_vectors = np.atleast_2d(lattice_vectors) # Turn into numpy array
            # Use the lattice vectors to do the basis transformation to the RL basis
            q = (1./(2.*np.pi)) * np.tensordot( lattice_vectors, k, axes=1 )
        
        if self.H=={}:
            self.coupling_matrices() # Compute coupling matrices
        
        ham = ham_momentum_RL(q, self.H_keys, self.H_vals, safety=safety) # Get the Bloch coefficient matrix
        
        if squeeze_output:
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
        
        print('self.H =')
        for key, val in self.H.items():
            print('\n{} -> \n{}'.format(key, val))
        print('*'*80)
        
        print(f'{self.classical_energy() = }')
        
        return


def test():
    np.set_printoptions(linewidth=250)
    
    dim = 2
    
    r0 = Rotation.identity()
    r1 = Rotation.from_rotvec(np.pi * np.array([0,1,0]))
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
