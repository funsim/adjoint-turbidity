import random
from dolfin import *
from dolfin_adjoint import *
from numpy import dot
import pylab

def test_gradient_array(J, dJ, x, seed=0.01, perturbation_direction=None, plot_file=None):
    '''Checks the correctness of the derivative dJ.
       x must be an array that specifies at which point in the parameter space
       the gradient is to be checked. The functions J(x) and dJ(x) must return
       the functional value and the functional derivative respectivaly.

       This function returns the order of convergence of the Taylor
       series remainder, which should be 2 if the gradient is correct.'''

    # We will compute the gradient of the functional with respect to the initial condition,
    # and check its correctness with the Taylor remainder convergence test.
    info_blue("Running Taylor remainder convergence analysis to check the gradient ... ")

    # First run the problem unperturbed
    info_blue("Running forward model ... ")
    j_direct = J(x)

    # obtain gradient
    info_blue("Running adjoint model ... ")
    dj = dJ(x, forget=True)

    # Randomise the perturbation direction:
    if perturbation_direction is None:
        perturbation_direction = x.copy()
        # Make sure that we use a consistent seed value accross all processors
        random.seed(243)
        for i in range(len(x)):
          perturbation_direction[i] = random.random()

    # Run the forward problem for various perturbed initial conditions
    functional_values = []
    perturbations = []
    perturbation_sizes = [seed / (2 ** i) for i in range(5)]
    for perturbation_size in perturbation_sizes:
        perturbation = perturbation_direction.copy() * perturbation_size
        perturbations.append(perturbation)

        perturbed_x = x.copy() + perturbation
        info_blue("Rerunning forward model with perturbation")
        functional_values.append(J(perturbed_x))

    # First-order Taylor remainders (not using adjoint)
    no_gradient = [abs(perturbed_j - j_direct) for perturbed_j in functional_values]

    info_green("dJ using gradient                 : %s" % str(['%+010.7e'%d[0] for d in dot(perturbations, dj)]))
    info_green("actual dJ                         : %s" % str(['%+010.7e'%(perturbed_j - j_direct) for perturbed_j in functional_values]))
    # info_green("Absolute functional evaluation differences: %s" % str(no_gradient))
    info_green("Convergence orders without adjoint: %s" % str(['%+010.7e'%g for g in convergence_order(no_gradient)]))

    with_gradient = []
    for i in range(len(perturbations)):
        remainder = abs(functional_values[i] - j_direct - dot(perturbations[i], dj))
        with_gradient.append(remainder)

    info_green("Functional evaluation differences : %s" % str(['%+010.7e'%g[0] for g in with_gradient]))
    info_green("Convergence orders with adjoint   : %s" % str(['%+010.7e'%g for g in convergence_order(with_gradient)]))

    if plot_file:
        first_order = [xx for xx in perturbation_sizes]
        second_order = [xx ** 2 for xx in perturbation_sizes]

        pylab.figure()
        pylab.loglog(perturbation_sizes, first_order, 'b--', perturbation_sizes, second_order, 'g--', perturbation_sizes, no_gradient, 'bo-', perturbation_sizes, with_gradient, 'go-')
        pylab.legend(('First order convergence', 'Second order convergence', 'Taylor remainder without gradient', 'Taylor remainder with gradient'), 'lower right', shadow=True, fancybox=True)
        pylab.xlabel("Perturbation size")
        pylab.ylabel("Taylor remainder")
        pylab.savefig(plot_file)

    return min(convergence_order(with_gradient))
