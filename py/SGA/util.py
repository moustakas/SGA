"""
SGA.util
========

Support utilities.

"""
import numpy as np

def match_to(A, B):
    """Return indexes where B matches A, holding A in place.

    Parameters
    ----------
    A : :class:`~numpy.ndarray` or `list`
        Array of integers to match TO.
    B : :class:`~numpy.ndarray` or `list`
        Array of integers matched to `A`

    Returns
    -------
    :class:`~numpy.ndarray`
        The integer indexes in A that B matches to.

    Notes
    -----
    - Result should be such that for ii = match_to(A, B)
      np.all(A[ii] == B) is True.
    - We're looking up locations of B in A so B can be
      a shorter array than A (provided the B->A matches are
      unique) but A cannot be a shorter array than B.
    """
    # ADM grab the integer matchers.
    Aii, Bii = match(A, B)

    # ADM return, restoring the original order.
    return Aii[np.argsort(Bii)]


def match(A, B):
    """Return matching indexes for two arrays on a unique key.

    Parameters
    ----------
    A : :class:`~numpy.ndarray` or `list`
        An array of integers.
    B : :class:`~numpy.ndarray` or `list`
        An array of integers.

    Returns
    -------
    :class:`~numpy.ndarray`
        The integer indexes in A to match to B.
    :class:`~numpy.ndarray`
        The integer indexes in B to match to A.

    Notes
    -----
    - Result should be such that for Aii, Bii = match(A, B)
      np.all(A[Aii] == B[Bii]) is True.
    - Only works if there is a unique mapping from A->B, i.e
      if A and B do NOT contain duplicates.
    - h/t Anand Raichoor `by way of Stack Overflow`_.
    """
    # AR sorting A,B
    tmpA = np.sort(A)
    tmpB = np.sort(B)

    # AR create mask equivalent to np.in1d(A,B) and np.in1d(B,A) for unique elements
    maskA = (
        np.searchsorted(tmpB, tmpA, "right") - np.searchsorted(tmpB, tmpA, "left")
    ) == 1
    maskB = (
        np.searchsorted(tmpA, tmpB, "right") - np.searchsorted(tmpA, tmpB, "left")
    ) == 1

    # AR to get back to original indexes
    return np.argsort(A)[maskA], np.argsort(B)[maskB]


def weighted_partition(weights, n):
    '''
    Partition `weights` into `n` groups with approximately same sum(weights)

    Args:
        weights: array-like weights
        n: number of groups

    Returns list of lists of indices of weights for each group

    Notes:
        compared to `dist_discrete_all`, this function allows non-contiguous
        items to be grouped together which allows better balancing.

    '''
    #- sumweights will track the sum of the weights that have been assigned
    #- to each group so far
    sumweights = np.zeros(n, dtype=float)

    #- Initialize list of lists of indices for each group
    groups = list()
    for i in range(n):
        groups.append(list())

    #- Assign items from highest weight to lowest weight, always assigning
    #- to whichever group currently has the fewest weights
    weights = np.asarray(weights)
    for i in np.argsort(-weights):
        j = np.argmin(sumweights)
        groups[j].append(i)
        sumweights[j] += weights[i]

    assert len(groups) == n

    return groups
