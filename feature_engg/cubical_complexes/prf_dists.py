import copy
import numpy as np
from homology import PRF

class NormalPRF:
	"""
	A persistence rank function scaled such that the area of its domain is 1.
	This makes the comparison of PRFs with different domains sensible. This is
	especially useful if a set of PRFs has been generated with
	``filt_params['max_filtration_param'] < 0 `` (that is, the filtrations end
	at the epsilon value where their first n-simplex appears.)

	Parameters
	----------
	prf: PRF or 2d array
		source prf

	Class Attributes
	----------------
	dom_area: int or float
		area of normal prf domain
	"""
	dom_area = 1        # area of PRF domain (the triangle)
	lim = np.sqrt(dom_area * 2)

	def __init__(self, prf):
		if isinstance(prf, PRF) or isinstance(prf, NormalPRF):
			self.data = prf.data
		elif isinstance(prf, np.ndarray):
			self.data = prf

		self.num_div = self.data.shape[0]
		self.epsilons = np.linspace(0, self.lim, self.num_div)
		self.weight = None
		self.pre_weight = self


	@property
	def norm(self):
		"""
		Returns
		-------
		float
			L2 norm

		"""
		dA = (NormalPRF.lim / self.num_div) ** 2
		b = np.power(np.nansum(np.square(self.data) * dA),.5)
		return b


	def set_weight(self, wf):
		"""
		applies weight function to :py:attribute:`data`

		Parameters
		----------
		wf: lambda
			weight function as a lambda function, eg ``lambda i, j: -i + j``
		"""
		if wf is None:
			return
		self.weight = wf
		self.pre_weight = copy.deepcopy(self)

		x = y = np.linspace(0, np.sqrt(self.dom_area * 2), self.num_div)
		xx, yy = np.meshgrid(x, y)
		wf_arr = wf(xx, yy)
		if isinstance(wf_arr, int):
			wf_arr = np.full_like(xx, wf_arr)
		self.data = np.multiply(self.data, wf_arr)


	@classmethod
	def mean(cls, nprfs):
		"""
		Parameters
		----------
		nprfs: array
			1d array of ``NormalPRF``s

		Returns
		-------
		NormalPRF
			pointwise mean of ``nprfs``

		"""
		raws = [nprf.data for nprf in nprfs]
		return cls(np.mean(raws, axis=0))


	@classmethod
	def var(cls, nprfs):
		"""
		Parameters
		----------
		nprfs: array
			1d array of ``NormalPRF``s

		Returns
		-------
		NormalPRF
			pointwise variance of ``nprfs``

		"""
		raws = [nprf.data for nprf in nprfs]
		return cls(np.var(raws, axis=0))


	@staticmethod
	def _interpret(other):
		if isinstance(other, NormalPRF):
			return other.data
		elif isinstance(other, (int, float, long, np.ndarray)):
			return other
		else:
			raise TypeError

	def __add__(self, other):
		return NormalPRF(self.data + self._interpret(other))

	def __sub__(self, other):
		return NormalPRF(self.data - self._interpret(other))

	def __mul__(self, other):
		return NormalPRF(self.data * self._interpret(other))

	def __div__(self, other):
		return NormalPRF(np.true_divide(self.data,self._interpret(other)))

	def __pow__(self, other):
		return NormalPRF(np.power(self.data, self._interpret(other)))


def consecutive_dists(funcs): 

    dists = [np.nan];

    for i in range(1, len(funcs)):
        if funcs[i] == -1 or funcs[i-1] == -1:
            dists.append(np.nan);
        else:
            dists.append((NormalPRF(funcs[i]) - NormalPRF(funcs[i-1])).norm);        
    
    dists = np.array(dists);
    return dists;
