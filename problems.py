from collections import Counter
import time
import theano
from theano import function
import random
from ipdb import set_trace as debug
import theano.tensor as T
import numpy as np

theano.config.compute_test_value = 'ignore'

#Returns a new function, which evaluates f and increments counter
#cost is the size of the increment
#counter will also track the cost incurred by functions with name `name'
#name defaults to f.__name__
#if no counter is specified, a new one is created and returned as the second value
def with_cost(f, counter=None, cost=1, name=None):
    if name is None:
        name = f.__name__
    new_counter = False
    if counter is None:
        new_counter = True
        counter = Counter()
    def f_with_cost(*args):
        pay_cost(counter, cost, name)
        return f(*args)
    if new_counter:
        return f_with_cost, counter
    else:
        return f_with_cost

def pay_cost(counter, cost=1, name=None):
    counter['total_cost'] += cost
    if name is not None:
        counter[name] += 1

def repeat_for_input(weights, inputs):
    return T.extra_ops.repeat(T.shape_padleft(weights), inputs.shape[0], 0)

def test(x):
    if theano.config.compute_test_value != 'off':
        return x.tag.test_value
    else:
        return None

def settest(x, value):
    if theano.config.compute_test_value != 'off':
        x.tag.test_value = value

def quadratic(A, b):
    x = theano.shared(np.zeros_like(b))
    Ax = T.constant(A).dot(x)
    loss = 0.5 * x.dot(Ax) - T.constant(b).dot(x)
    model = {
        'params':[x],
        'loss':loss,
        'outputs':x,
        'data':[],
        'grad2_from_model':False,
    }
    return model

#TODO: I should somehow cache the code for the logistic model rather than recomputing it every time
def logistic_model(N=None, K=None, initial_weights = None, variance = 1.0, regularization = 0.0):
    if (not N or not K) and not initial_weights:
        raise Exception("Logistic model requires either N, K or initial_weights")
    if N is None:
        N, K = np.shape(initial_weights)
    elif initial_weights is None:
        initial_weights = variance * np.random.randn(N, K) / np.sqrt(N)

    #This is the number of instances to have for the test_values, used in debuggin
    M = 10

    weights = theano.shared(initial_weights)
    bias = theano.shared(np.zeros(K))

    inputs = T.matrix('inputs')
    settest(inputs, np.random.randn(M,N))


    targets = T.matrix('targets')
    settest(targets, np.ones((M, K)) * (1.0 / K))

    outputs = T.nnet.softmax(bias + T.dot(inputs,weights))
    loss = T.nnet.categorical_crossentropy(outputs, targets).mean() + \
            regularization * (weights * weights).sum()

    #This is an annoying way to efficiently compute the jacobian of losses w.r.t. parameters.
    #We copy each parameter once for each instance, and then compute gradients
    #with respect to the copied parameters. We can then e.g. square these gradients and sum them.
    #Directly computing the jacobian of losses against parameters takes too long in theano.
    #NOTE: This is going to cause trouble when we try to compute Gv products,
    #unless Theano has been modified so taht Repeat supports Rop (my local copy has)
    repeated_weights = repeat_for_input(weights, inputs) 
    repeated_bias = repeat_for_input(bias, inputs)

#NOTE: repeated_outputs is the same as outputs, but computed in an inefficient way that
#tracks the dependencies on different inputs
    repeated_outputs = T.nnet.softmax(repeated_bias + T.batched_dot(inputs,repeated_weights))
    total_loss_for_grad2 = T.nnet.categorical_crossentropy(repeated_outputs, targets).sum() + \
            regularization * (repeated_weights * repeated_weights).sum()

    model = {
        'inputs':inputs,
        'targets':targets,
        'data':[inputs, targets],
        'outputs':outputs,
        'params':[weights, bias],
        'repeated_params':[repeated_weights, repeated_bias],
        'grad2_from_model':True,
        'total_loss_for_grad2':total_loss_for_grad2,
        'loss':loss
    }
    return model

def symbolic_version(vs):
    result = []
    for v in vs:
        symbolic_v = T.TensorType(
            dtype=np.dtype('float64'), broadcastable=v.broadcastable
        )()
        settest(symbolic_v, v.get_value())
        result.append(symbolic_v)
    return result

#TODO: should let model choose whether to use Gv or Hv
#TODO: should split out the minibatch management and so on into a separate class

class OptimizationProblem(object):

    #NOTE: minibatch size is currently unused
    def __init__(self, name, model, data, minibatch_size=100):
        self.data = list(data) #inputs, targets
        self.model = model
        model = self.model
        self.name = name

        self.params = self.model['params']
        self.original_param_values = [ param.get_value() for param in self.params ]
        self.param_shapes = [ value.shape for value in self.original_param_values ]
        self.param_sizes = map(lambda x : int(np.prod(x)), self.param_shapes)
        self.param_positions = np.cumsum([0] + self.param_sizes)[:-1]
        self.zeros = np.zeros(np.sum(self.param_sizes))
        param_vs = symbolic_version(self.params)#creates list of symbolic versions of params

        offsets = symbolic_version(self.params)
        offset_givens = [ (param, param + offset) 
                for param, offset 
                in zip(self.params, offsets) ]
        #when these variables are non-zero, they modify the model parameters prior to evaluation

        self.cost_counter = Counter() #count function calls

        self.gradient_f = with_cost(
            function(self.model['data'] + offsets, T.grad( model['loss'], model['params'] ),
                on_unused_input='ignore', givens = offset_givens),
            counter = self.cost_counter, 
            cost = 1,
            name='gradient'
        )

        #TODO: compute grad2 and gradient at the same time...
        #TODO: save on memory for grad2 (make sure theano doesn't store intermediate results)
        #note: total_loss is just the sum of losses over instances, rather than average
        if model.get('grad2_from_model', False):
            all_gradients = T.grad( model['total_loss_for_grad2'], model['repeated_params'] )
            self.grad2_f = with_cost(
                function(self.model['data']+ offsets, [(g*g).mean(0) for g in all_gradients],
                    on_unused_input='ignore', givens = offset_givens,profile=True),
                counter = self.cost_counter,
                cost=1,
                name='gradient'
            )

        Jv = T.Rop(model['outputs'], model['params'],param_vs)
        HJv = T.grad(T.sum(T.grad(model['loss'], model['outputs'])*Jv), model['outputs'],
                consider_constant=[Jv],
                disconnected_inputs='ignore'
              )
        Gv = T.grad(T.sum(HJv * model['outputs']), model['params'],
                consider_constant=[HJv, Jv],
                disconnected_inputs='ignore'
            )
        self.Gv_f = with_cost(
            function(self.model['data'] + param_vs + offsets, Gv, 
                on_unused_input='ignore', givens = offset_givens),
            counter = self.cost_counter,
            cost = 2,
            name='Gv'
        )
        
        self.objective_f = with_cost(
            function(self.model['data'] + offsets, model['loss'],
                givens = offset_givens),
            counter = self.cost_counter,
            cost = 1,
            name='objective'
        )

    def minibatch(self, **kwargs):
        return self.minibatches(n=1,**kwargs).next()

#TODO: if n = None include everything, don't round it down
    def minibatches(self, n=None, size=None):
        if size is None:
            size = self.minibatch_size
        dataset_size = len(self.data[0])
        if n is None:
            n = dataset_size // size
        indices  = random.sample(xrange(dataset_size), n*size) 
        for i in range(n):
            yield [ datum[indices[i*n:(i+1)*n]] for datum in data ]

    def render_params(self):
        pass

    def reset(self):
        self.cost_counter.clear()
        for param, value in zip(self.params, self.original_param_values):
            param.set_value(value)

    def __repr__(self):
        d = self.render_params()
        return self.name + ((":" + d) if d else "")

    def current_solution(self):
        return self.pack_values( [param.get_value() for param in self.params] )

    def pay_cost(self, **kwargs):
        return pay_cost(self.cost_counter, **kwargs)

    def cost(self):
        return self.cost_counter['total_cost']

    def update(self, offsets):
        unpacked_offsets = self.unpack_values(offsets)
        for param, offset in zip(self.params, unpacked_offsets):
            param.set_value(param.get_value() + offset)

    #TODO: fix this copy-pasted code: [grad2, gradient, Gv, and objective are all very similar]
    def grad2(self, offset=None):
        if self.model.get('grad2_from_model', False):
            if offset is None:
                offset = self.zeros
            unpacked_grad2 = self.grad2_f(*(self.data + self.unpack_values(offset)))
            return self.pack_values(unpacked_grad2)
        else:
            raise Exception("If you don't compute grad2_from_model, you should implement grad2.")

    def gradient(self, offset=None):
        if offset is None:
            offset = self.zeros
        unpacked_gradient = self.gradient_f(*(self.data + self.unpack_values(offset)))
        return self.pack_values(unpacked_gradient)

    def Gv(self, v, offset=None):
        if offset is None:
            offset = self.zeros
        unpacked_Gv = self.Gv_f(*(self.data + self.unpack_values(v) + self.unpack_values(offset)))
        return self.pack_values(unpacked_Gv)

    def objective(self, offset=None):
        if offset is None:
            offset = self.zeros
        return self.objective_f(*(self.data + self.unpack_values(offset)))

    def unpack_values(self, value_vector):
        return [ value_vector[position:position+size].reshape(shape) for shape,size,position in zip(self.param_shapes, self.param_sizes, self.param_positions) ]

    def pack_values(self, value_list):
        return np.concatenate([value.flatten() for value in value_list])

class Quadratic(OptimizationProblem):
    def __init__(self, A, b):
        super(Quadratic, self).__init__('Quadratic', quadratic(A, b), [])
        self.A = A
        self.b = b
        self.optimum = self.objective(np.linalg.inv(A).dot(b))

    def render_params(self):
        return "N={}".format(len(self.b))

    def grad2(self, offset=None):
        self.pay_cost()
        d = np.diag(self.A)
        return d


class RandomQuadratic(Quadratic):

    def __init__(self, N=3):
        rA = np.random.randn(N,N)
        A = rA.dot(rA.transpose())
        b = np.random.randn(N)
        super(RandomQuadratic, self).__init__(A, b)

def onehot(k,K):
    result = np.zeros(K)
    result[k] = 1.0
    return result

class OrganicRegressionProblem(OptimizationProblem):

    def __init__(self, inputs, outputs, name):
        N = len(inputs[0])
        if type(outputs[0]) is int:
            K = max(outputs)+1
            outputs = [onehot(k,K) for k in outputs]
        K = len(outputs[0])
        model = logistic_model(N=N,K=K)
        data = np.array(inputs), np.array(outputs)
        super(OrganicRegressionProblem, self).__init__("LogisticRegression:{}".format(name), model, data)
        self.inputs = inputs
        self.outputs = outputs

    def render_params(self):
        return "{},M={}".format(self.name, self.M)


class SyntheticRegressionProblem(OptimizationProblem):

    def __init__(self, N, M, K=2):
        test_model = logistic_model(N=N, K=K)
        real_model = logistic_model(N=N, K=K)
        inputs = np.random.randn(M, N)
        f = function([real_model['inputs']], real_model['outputs'])
        data = [inputs, f(inputs)]
        super(SyntheticRegressionProblem, self).__init__("LogisticRegression", test_model, data)
        self.optimum = np.mean([ sum(x*np.log(x) for x in datum) for datum in self.data[1] ])
        self.N = N
        self.M = M
        self.K = K

    def render_params(self):
        return "N={},K={},M={}".format(self.N,self.K,self.M)
