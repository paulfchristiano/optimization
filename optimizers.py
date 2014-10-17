from damped_cg import damped_cg
from collections import Counter
import time
import theano
from theano import function
from ipdb import set_trace as debug
import theano.tensor as T
import numpy as np

theano.config.compute_test_value = 'ignore'

#TODO: allow each optimizer to return a list of its important params, to be used for reporting during the optimization
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

    def __init__(self, name):
        self.name = name
        self.reset()
        pass

    def __repr__(self):
        d = self.render_params()
        return self.name + ((":" + d) if d else "")

    def render_params(self):
        return ""

    def reset(self):
        self.iterations = 0
        pass

    def reset_with_P(self, P):
        pass

    def step(self, P):
        if self.iterations == 0:
            self.reset_with_P(P)
        self.iterations += 1
        pass

    def should_terminate(self, values, tolerance, testgap):
        if len(values) > testgap:
            total_improvement = values[0] - values[-1]
            recent_improvement = values[-1-testgap] - values[-1]
            return total_improvement <= 0 or recent_improvement < tolerance*total_improvement
        else:
            return False

    def status_report(self, values, last_cost):
        print("[Iteration {}: cost = {}, values = {}]".format(self.iterations, last_cost, values[-1]))
        print(self.__dict__)
        print("t = {}".format(time.time() - self.start_time))

    def optimize(self, P, tolerance=1e-3, testgap=4, interval=100, max_cost=20000):
        self.start_time = time.time()
        values = [P.objective()]
        last_cost = P.cost()
        while True:
            self.step(P)
            if P.cost() > last_cost + interval:
                last_cost = P.cost()
                values.append(P.objective())
                self.status_report(values, last_cost)
                if self.should_terminate(values, tolerance, testgap) or last_cost>max_cost:
                    return values[-1], last_cost

class GradientOptimizer(Optimizer):

    #This method is called when the optimizer is initialized
    #should call super with the name of the optimizer, for printing purposes
    #should set all of the parameters
    def __init__(self,
            learning_rate = 1e-2
        ):
        super(GradientOptimizer, self).__init__("GradientDescent")
        self.learning_rate = learning_rate

    #This method renders a description of the optimizer's parameters
    #again, this is just for printing purposes
    def render_params(self):
        return repr(self.learning_rate)

    #this performs one step of optimization
    #it should call super to do things like increment the number of steps taken
    #it can make use of P.gradient, P.objective, which calculate the gradient and objective
    #at the current point. P.update(d) adjusts the current point by adding d
    #if you call P.objective(d) or P.gradient(d), it calculates the gradient
    #or objective at an offset of d from the current point.
    def step(self, P):
        super(GradientOptimizer, self).step(P)
        g = -P.gradient()
        P.update(g * self.learning_rate)

    #you can also write a reset method to reset any internal state of the optimizer
    #or reset_with_P to reset internal state in a way that depends on the problem P

class MomentumOptimizer(Optimizer):

    def __init__(self, momentum=0.9, rate = 1e-3):
        super(MomentumOptimizer, self).__init__("Momentum")
        self.momentum = momentum
        self.learning_rate = rate

    def render_params(self):
        return "{},m={}".format(self.learning_rate, self.momentum)

    def reset(self):
        super(MomentumOptimizer, self).reset()
        self.inertia = 0

    def step(self, P):
        super(MomentumOptimizer, self).step(P)
        g = -P.gradient()
        self.inertia = g + self.momentum*self.inertia
        P.update(self.inertia * self.learning_rate)
        

class AdaGradOptimizer(Optimizer):

    def __init__(self, learning_rate = 1e-2):
        self.learning_rate = learning_rate
        super(AdaGradOptimizer, self).__init__("AdaGrad")

    def render_params(self):
        return repr(self.learning_rate)

    def reset(self):
        super(AdaGradOptimizer, self).reset()
        self.total_grad = 1e-10

    def step(self, P):
        super(AdaGradOptimizer, self).step(self)
        g = -P.gradient()
        self.total_grad += g*g
        P.update(self.learning_rate * g / np.sqrt(self.total_grad) )

class HFOptimizer(Optimizer):

    def __init__(self, 
                initial_lambda = 0.01,
                initial_min_iters = 10
            ):
        self.lambda_multipliers = [1.0, 0.9, 1.1, 0.8, 1.2, 0.5, 1.5, 2.0, 0.1, 10.0 ]
        self.initial_lambda = initial_lambda
        self.initial_min_iters = initial_min_iters
        super(HFOptimizer, self).__init__("HF")

    def reset(self):
        super(HFOptimizer, self).reset()
        self.current_lambda = self.initial_lambda
        self.min_iters = self.initial_min_iters

    def reset_with_P(self, P):
        self.x = P.zeros

    def step(self, P):
        super(HFOptimizer, self).step(P)

        grad2 = P.grad2()
        preconditioner = (grad2 + 0.01*grad2.mean()*np.ones_like(grad2))**(0.75)

        x_best, i_best, m_best, x_last, obj_best = damped_cg(
            P.Gv, -P.gradient(), self.x,
            P.objective, preconditioner, self.current_lambda,
            [( multiplier - 1) * self.current_lambda for multiplier in self.lambda_multipliers],
            subspace_size=5
        )

        self.min_iters = i_best
        self.x = x_last
        self.current_lambda *= self.lambda_multipliers[m_best]
        
        P.update(x_best)
            
#Computation of Gv product from https://github.com/boulanni/theano-hf
#(Along with parameter flattening and unflattening, etc.)


