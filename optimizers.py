import time
from copy import copy

from ipdb import set_trace as debug
import numpy as np
import theano
import theano.tensor as T
from theano import function

from damped_cg import banded_cg,augmented_cg
from old_damped_cg import augmented_cg as old_augmented_cg
from recording import Record
from linesearch import backtracking_linesearch

theano.config.compute_test_value = 'ignore'

def arsuper(c, instance):
    instance.__class__ = globals()[instance.__class__.__name__]
    return super(c, instance)

#TODO: change the step methods to record whatever data you'd want to look through
#TODO: test all of this and make it actually work

#TODO: update/move/deprecate these printing methods
def padstring(s, n):
    m = len(s)
    leftpadding = (n-m)/ 2
    rightpadding = (n-m+1)/2
    return (leftpadding * " " ) + s + (rightpadding * " ")

def nice_repr(s):
    return s if type(s) is str else repr(s)

def render_table(columns, rows, table):
    left_border_size = max( len(nice_repr(row)) for row in rows )
    cells = [ [""] + [nice_repr(column) for column in columns ] ]
    column_widths = [len(val) for val in cells[0]]
    for row, table_row in zip(rows, table):
        cells.append([nice_repr(row)] + [t for t in table_row])
        for j, val in enumerate(cells[-1]):
            column_widths[j] = max(column_widths[j], len(val))
    row_seperator = "\n" + "|".join( "-"*width for width in column_widths ) + "\n"
    return row_seperator.join(
        "|".join( 
            padstring(cell, width) for cell, width in zip (row, column_widths)
        ) for row in cells
    )


def test_on(optimizer, problem):
    optimizer.reset()
    problem.reset()
    value, cost = optimizer.optimize(problem)
    return "{} \ {}".format(value, cost)

def test_all(optimizers, problems):
    results = [ [test_on(optimizer, problem) for optimizer in optimizers ] 
            for problem in problems]
    print(render_table(optimizers, problems, results))
    return results

class Optimizer(object):

    def __init__(self, name, note=None):
        self.name = name
        self.debugmode=False
        self.note = note
        self.monitors = set()
        self.records = []
        self.problem = None
        self.value = None

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

    def __repr__(self):
        d = self.render_params()
        return self.name \
                + ("[{}]".format(self.note) if self.note else "") \
                + ((":" + d) if d else "")

    def render_params(self):
        return ""

    def start(self, P, **kwargs):
        self.problem = P
        self.record = Record(self)
        self.value = self.initialization()
        self.reset()
        self.optimize(**kwargs)

    def resume(self, P, **kwargs):
        assert(P == self.problem)
        self.optimize(**kwargs)

    def initialization(self):
        return copy(self.problem.initial_value)

    def reset(self):
        pass

    def step(self):
        if self.debugmode:
            debug()
        self.record.iteration += 1

    def should_terminate(self, values, tolerance, testgap):
        if len(values) > testgap:
            total_improvement = values[0] - values[-1]
            recent_improvement = values[-1-testgap] - values[-1]
            return total_improvement <= 0 or recent_improvement < tolerance*total_improvement
        else:
            return False

    def status(self):
        return {param : self.__dict__[param] for param in self.monitors}

    def grad(self, x, **kwargs):
        return self.problem.grad(x, record=self.record, **kwargs)

    def Hv(self, x, v, **kwargs):
        return self.problem.Hv(x, v, record=self.record, **kwargs)

    def Gv(self, x, v, **kwargs):
        return self.problem.Gv(x, v, record=self.record, **kwargs)

    def grad2(self, x, **kwargs):
        return self.problem.grad2(x, record=self.record, **kwargs)

    def objective(self, x, **kwargs):
        return self.problem.objective(x, record=self.record, **kwargs)

    def snapshot(self, **kwargs):
        args = self.status()
        args.update(kwargs)
        if 'value' not in args:
            args['value'] = self.value
        return self.record.snapshot(**args)

    #TODO: better management of optimization tolerance etc., right now hard to change
    #this method isn't supposed to be called directly by the user
    def optimize(self, tolerance=1e-3, testgap=4, epoch_gap=100, max_epochs=50000):
        try:
            self.record.start_timer()
            starting_epoch = self.record.epoch()
            last_snapshot = -float('inf')
            values = []
            while True:
                if self.record.epoch() - last_snapshot >= epoch_gap:
                    last_snapshot = self.record.epoch()
                    values.append(self.snapshot())
                    if self.should_terminate(values, tolerance, testgap) \
                            or last_snapshot-starting_epoch>max_epochs:
                        return
                self.record.log(self.step())
        finally:
            self.record.stop_timer()

class GradientOptimizer(Optimizer):

    #This method is called when the optimizer is initialized
    #should call super with the name of the optimizer, for printing purposes
    #should set all of the parameters
    def __init__(self,
            learning_rate=1e-2,
            minibatch_size=None,
            **kwargs
        ):
        arsuper(GradientOptimizer, self).__init__("GradientDescent")
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size

    #This method renders a description of the optimizer's parameters
    #again, this is just for printing purposes
    def render_params(self):
        result = "{}".format(self.learning_rate)
        if self.minibatch_size is not None:
            result += " [{}]".format(self.minibatch_size)
        return repr(self.learning_rate)

    #this performs one step of optimization
    #it should call super to do things like increment the number of steps taken
    #it can make use of self.grad, self.objective, which calculate the gradient and objective
    #at the current point.
    def step(self):
        arsuper(GradientOptimizer, self).step()
        g = -self.grad(self.value,size=self.minibatch_size)
        self.value += g * self.learning_rate
        return [g.dot(g)]


class MomentumOptimizer(Optimizer):

    def __init__(self, momentum=0.9, rate = 1e-3, name="Momentum", **kwargs):
        arsuper(MomentumOptimizer, self).__init__(name, **kwargs)
        self.momentum = momentum
        self.learning_rate = rate
        self.monitors.add("velocity")

    def render_params(self):
        return "{},m={}".format(self.learning_rate, self.momentum)

    def reset(self):
        arsuper(MomentumOptimizer, self).reset()
        self.velocity = self.problem.zeros

    def step(self):
        arsuper(MomentumOptimizer, self).step()
        g = -self.grad(self.value)
        self.velocity = g*self.learning_rate + self.momentum*self.velocity
        self.value += self.velocity
        return [(self.velocity.dot(self.velocity), g.dot(g))]


class NAGOptimizer(MomentumOptimizer):
    
    def __init__(self, momentum=0.999, rate=1e-3, name="Accelerated Gradient", **kwargs):
        arsuper(NAGOptimizer, self).__init__(momentum, rate, name, **kwargs)

    def step(self):
        arsuper(Optimizer, self).step()
        g = -self.grad(self.value + self.velocity)
        self.velocity = g * self.learning_rate + self.momentum*self.velocity
        self.value += self.velocity
        return [(g.dot(g), self.velocity.dot(self.velocity))]


class AdaGradOptimizer(Optimizer):

    def __init__(self, learning_rate = 1e-2):
        self.learning_rate = learning_rate
        arsuper(AdaGradOptimizer, self).__init__("AdaGrad")
        self.monitors.add("total_grad")

    def render_params(self):
        return repr(self.learning_rate)

    def reset(self):
        arsuper(AdaGradOptimizer, self).reset()
        self.total_grad = 1e-10

    def step(self):
        arsuper(AdaGradOptimizer, self).step()
        g = -self.grad()
        self.total_grad += g*g
        self.value += self.learning_rate * g/ np.sqrt(self.total_grad)
        return [(self.total_grad.mean(), g.dot(g))]

class HFOptimizer_MultiL(Optimizer):

    #TODO: encapsulate the "general" MultiL settings in an appropriate object (if helpful)
    def __init__(self, 
                initial_lambda = 0.01,
                initial_min_iters = 10,
                imult=1.3,
                lambda_multipliers = None,
                bail=False,
                use_last = False,
                name="HFMultiL",
                **kwargs
            ):
        self.imult = imult
        self.use_last = use_last
        self.lambda_multipliers = lambda_multipliers if lambda_multipliers else \
            [0.1, 0.3, 0.6, 0.8,  0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 4.0, 10.0]
        self.initial_lambda = initial_lambda
        self.current_lambda = self.initial_lambda
        self.initial_min_iters = initial_min_iters
        self.bail = bail
        arsuper(HFOptimizer_MultiL, self).__init__(name, **kwargs)
        self.monitors.update(["current_lambda", "min_iters", "x"])

    def reset(self):
        arsuper(HFOptimizer_MultiL, self).reset()
        self.current_lambda = self.initial_lambda
        self.min_iters = self.initial_min_iters
        self.x = self.problem.zeros

    def CGstep(self, **kwargs):
        raise NotImplementedError()

    def step(self):
        arsuper(HFOptimizer_MultiL, self).step()

        grad2 = self.grad2(self.value)
        preconditioner = (grad2 + 0.01*grad2.mean()*np.ones_like(grad2))**(0.75)

        x_best, i_best, m_best, x_last, m_last, obj_best, logs = self.CGstep(
            lambda v : self.Gv(self.value, v), -self.grad(self.value), self.x,
            lambda d : self.objective(self.value+d), 
            [ multiplier  * self.current_lambda for multiplier in self.lambda_multipliers],
            preconditioner
        )

        self.min_iters = i_best
        self.x = x_last
        self.current_lambda *= self.lambda_multipliers[m_last if self.use_last else m_best]

        self.value += x_best

        return logs


class BandedHFOptimizer(HFOptimizer_MultiL):
    
    def __init__(self, name="BandedHF", **kwargs):
        arsuper(BandedHFOptimizer, self).__init__(name=name,**kwargs)

    def CGstep(self, apply_A, g, x0, objective, lambdas, preconditioner):
        return banded_cg(
            apply_A, g, x0, objective, preconditioner, self.current_lambda,
            lambdas, bail=self.bail, imult = self.imult
        )

class AugmentedHFOptimizer(HFOptimizer_MultiL):

    def __init__(self, name="AugmentedHF",subspace_size=5,old=False,**kwargs):
        arsuper(AugmentedHFOptimizer, self).__init__(name=name,**kwargs)
        self.subspace_size = subspace_size
        self.old = old

    def CGstep(self, apply_A, g, x0, objective, lambdas, preconditioner):
        return augmented_cg(
            apply_A, g, x0, objective, preconditioner, self.current_lambda,
            lambdas, subspace_size=self.subspace_size, bail=self.bail, imult = self.imult
        )

class LBFGS(Optimizer):

    def __init__(self, name="L-BFGS", m=10, **kwargs):
        arsuper(LBFGS, self).__init__(name=name, **kwargs)
        self.m = m

    def add_to_history(self, s, y):
        rho = 1 / s.dot(y)
        self.ss.append(copy(s))
        self.ys.append(copy(y))
        self.rhos.append(copy(rho))
        for l in [self.ss, self.ys, self.rhos]:
            if len(l) > self.m:
                del l[0]

    def reset(self):
        self.rhos = []
        self.ss = []
        self.ys = []
        self.g = None
        #TODO: improve the initial approximation
        self.Hinv0 = 1

    #Taken from homes.cs.washington.edu/~galen/files/quasi-newton-notes.pdf
    def Hinv(self, v):
        m = min(self.m, len(self.rhos))
        q = v
        alphas = np.zeros((m, len(v)))
        for i in range(m-1, -1, -1):
            alphas[i] = self.rhos[i] * q.dot(self.ss[i])
            q = q - alphas[i] * self.ys[i]
        r = np.dot(self.Hinv0, q)
        for i in range(0, m, +1):
            beta = self.rhos[i] * np.dot(self.ys[i], r)
            r = r + self.ss[i] * (alphas[i] - beta)
        return r

    def step(self):
        arsuper(LBFGS, self).step()
        if self.g is None:
            self.g = self.grad(self.value)

        direction = self.Hinv(-self.g)
        s = direction * backtracking_linesearch(
                f=self.objective,
                x=self.value, 
                p=direction,
                fx=self.objective(self.value),
                gx = self.g
            )

        new_value = self.value+s
        new_g = self.grad(new_value)
        y = new_g - self.g
        self.add_to_history(s, y)

        self.value = new_value
        self.g = new_g 
