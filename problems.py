from collections import Counter
import time
import random
from copy import copy
from ipdb import set_trace as debug
import array

import numpy as np
import theano
import theano.tensor as T
from theano import function

theano.config.compute_test_value = 'ignore'

def arsuper(c, instance):
    instance.__class__ = globals()[instance.__class__.__name__]
    return super(c, instance)

def test(x):
    if theano.config.compute_test_value != 'off':
        return x.tag.test_value
    else:
        return None

def settest(x, value):
    if theano.config.compute_test_value != 'off':
        x.tag.test_value = value

class OptimizationProblem(object):

    def __init__(self, name="OptimizationProblem", template=None):

        self.name = name
        self.records = []
        self.epoch_length = 1

    def test_all(self, optimizers):
        for optimizer in optimizers:
            optimizer.start(self)

    def objective(self, x):
        raise NotImplementedError()

    def grad(self, x):
        raise NotImplementedError()

    def grad2(self, x):
        raise NotImplementedError()

    def Hv(self, x, v):
        raise NotImplementedError()

    def Gv(self, x, v):
        raise NotImplementedError()

    def render_params(self):
        return None

    def __repr__(self):
        d = self.render_params()
        return self.name + ((":" + d) if d else "")

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

class TheanoOptimizationProblem(OptimizationProblem):

    def __init__(self, symbolic_params, params, objective, symbolic_inputs=[], inputs=[], **kwargs):
        arsuper(TheanoOptimizationProblem, self).__init__(**kwargs)
        self.symbolic_params = symbolic_params

        self.param_shapes = [ value.shape for value in params ]
        self.param_sizes = map(lambda x : int(np.prod(x)), self.param_shapes)
        self.param_positions = np.cumsum([0] + self.param_sizes)[:-1]

        self.inputs = [ np.array(x) for x in inputs ]
        self.epoch_length = 1 if not self.inputs else len(self.inputs[0])

        self.zeros = np.zeros(np.sum(self.param_sizes))
        self.initial_value = self.pack_values(params)

        #TODO: use givens+shared values for performance
        #i.e., allow the user to either use the shared value (already on GPU) or supply a new variable

        #TODO: think about the handling of regularizers

        grad = T.grad( objective.mean(), symbolic_params )

        self.grad_f = {'fn': function(symbolic_params+symbolic_inputs, grad, on_unused_input='ignore'),
            'cost':1, 'name':'grad'
        }

        #using T.arange(max_inputs) is an (endorsed) hack
        #T.arange needs its argument at compile-time
        #we include objective as a dummy argument, and scan will truncate
        #T.arange to the length of the shortest argument (i.e., the number of points)
        max_inputs=1000000
        #TODO: this is horrendously slow in the current implementation of Theano
        grad2 = [ theano.scan(fn = lambda dummy, i, prior_result, objective : 
                    prior_result + T.grad(objective[i], param)**2,
                    outputs_info=T.zeros_like(param),
                    sequences=[objective, T.arange(max_inputs)],
                    non_sequences=[objective]
                  )[0][-1] / (objective.shape[0])  for param in symbolic_params ]

        self.grad2_f = {'fn': function(symbolic_params+symbolic_inputs, grad2, on_unused_input='ignore'),
            'cost':1, 'name':'grad2'
        }

        self.objective_f = { 'fn': function(symbolic_params+symbolic_inputs, objective.mean()),
            'cost':1, 'name':'objective'
        }

        #TODO: implement Hv 
        #(main problem is Rop availability in Theano, could catch notimplementerrors)

        self.minibatches={}

    #TODO: allow two different optimizers to use the same problem, tracking their own minibatches
    #TODO: probably fix this at the same time as makign the batches thing reasonably performant
    def clear_batches(self,name=None):
        if name is None:
            self.minibatches = {}
        elif name in self.minibatches:
            del self.minibatches[name]

    def make_minibatch(self,size=None, reuse=False, name=None):
        #TODO: get rid of the x[indices] call for performance
        if not self.inputs:
            return []
        n = len(self.inputs[0])
        if size is None or size == n:
            return self.inputs
        if reuse and name in self.minibatches:
            return self.minibatches[name]
        indices = random.sample(xrange(n), size)
        result = [ x[indices] for x in self.inputs ]
        self.minibatches[name]=result
        return result

    def execute(self, f, xs, record=None, pack=True, unpacks=None, **kwargs):
        if unpacks is None:
            unpacks = [ True for x in xs ] 
        if xs and len(xs[0].shape) > 1 and xs[0].shape[0] > 1:
            kwargs['size'] = xs[0].shape[0]
        inputs = self.make_minibatch(**kwargs)
        size = 1 if not inputs else len(inputs[0])
        if record is not None:
            record.pay_cost(size * f['cost'], f.get('name',None))
        args = []
        for x, unpack in zip(xs, unpacks):
            if len(x.shape) == 1:
                x = np.repeat(x.reshape(1, -1), size, axis=0) 
            next_arg = self.unpack_values(x) if unpack else x
            args.extend(next_arg)
        result =  f['fn'](*(args + inputs))
        return self.pack_values(result) if pack else result

    def grad2(self, x, **kwargs):
        if 'size' not in kwargs:
            kwargs['size'] = 20
            return sum(self.execute(self.grad2_f, [x], **kwargs) for i in range(500)) / 500
        return self.execute(self.grad2_f, [x], **kwargs)

    def grad(self, x, **kwargs):
        return self.execute(self.grad_f, [x], **kwargs)

    def objective(self, x, **kwargs):
        return self.execute(self.objective_f, [x], pack=False, **kwargs)

    def unpack_values(self, value_vector):
        stacked_values = len(value_vector.shape) > 1
        if stacked_values:
            num_values = value_vector.shape[0]
        return [ value_vector[:, position:position+size].reshape((num_values,)+shape) 
                if stacked_values 
                else value_vector[position:position+size].reshape((1,)+ shape)
                for shape,size,position 
                in zip(self.param_shapes, self.param_sizes, self.param_positions) ]

    def pack_values(self, value_list):
        stacked_values = len(value_list[0].shape) > len(self.param_shapes[0])
        if stacked_values:
            num_values = value_list[0].shape[0]
            return np.concatenate([value.reshape(num_values, -1) for value in value_list], axis=1)
        else:
            return np.concatenate([value.flatten() for value in value_list])


class Quadratic(TheanoOptimizationProblem):

    def __init__(self, A, b, name="Quadratic"):
        self.N = len(b)
        x = T.vector('x')
        Ax = T.constant(A).dot(x)
        objective = 0.5 * x.dot(Ax) - T.constant(b).dot(x)
        self.A = A
        arsuper(Quadratic, self).__init__(
                symbolic_params = [x],
                params = [np.zeros_like(b)],
                objective = objective,
                symbolic_inputs = [],
                inputs = [],
                name=name
            )

    def render_params(self):
        return "N={}".format(self.N)

    def grad2(self):
        self.pay_cost(1, "grad2")
        return np.diag(self.A)

class RandomQuadratic(Quadratic):

    def __init__(self, N=3):
        rA = np.random.randn(N,N)
        A = rA.dot(rA.transpose())
        b = np.random.randn(N)
        arsuper(RandomQuadratic, self).__init__(A, b)


class PredictionProblem(TheanoOptimizationProblem):

    def __init__(self, symbolic_params, params, prediction, objective, symbolic_features, symbolic_targets, inputs, **kwargs):
        symbolic_inputs = symbolic_features + [symbolic_targets]
        arsuper(PredictionProblem, self).__init__(
                symbolic_params, params, objective, 
                symbolic_inputs, inputs, **kwargs)

        v = [T.copy(param) for param in symbolic_params]
        self.prediction = prediction

        #ATTRIBUTION: computation of Gv is from https://github.com/boulanni/theano-hf,
        #due to Nicolas Boulanger-Lewandowski,
        #along with code for paramter flattening and unflattening
        Jv = T.Rop(prediction, symbolic_params, v)
        HJv = T.grad(T.sum(T.grad(objective.mean(), prediction)*Jv), prediction,
                consider_constant=[Jv],
                disconnected_inputs='ignore'
              )
        Gv = T.grad(T.sum(HJv * prediction),symbolic_params,
                consider_constant=[HJv, Jv],
                disconnected_inputs='ignore'
            )
        self.Gv_f = {'fn': function(symbolic_params+v+symbolic_inputs, Gv, on_unused_input='ignore'),
            'cost':2, 'name':'Gv'
        }

        self.predictor = function(symbolic_params+symbolic_features,prediction)

    def Gv(self, x, v, **kwargs):
        return self.execute(self.Gv_f, [x, v], **kwargs)

    def predict(self, x, features):
        return self.predictor(*(self.unpack_values(x) + features))
                     

#TODO: fix this...
class Quadratic(TheanoOptimizationProblem):
    def __init__(self, A, b):
        self.A = A
        self.b = b
        arsuper(Quadratic, self).__init__('Quadratic', quadratic(A, b), [])
        self.optimum = self.objective(np.linalg.inv(A).dot(b))

    def render_params(self):
        return "N={}".format(len(self.b))

    def grad2(self, offset=None):
        self.pay_cost()
        d = np.diag(self.A)
        return d


class LogisticRegressionProblem(PredictionProblem):

    def __init__(self, features, targets, name="LogisticRegression", **kwargs):

        self.M = len(features)
        self.N = len(features[0])
        self.K = max(targets)+1

        unnormalized_features = np.array(features)
        normalized_features = features / (np.max(np.abs(features), axis=0) + 1e-6)
        
        symbolic_features = T.matrix('features')
        symbolic_targets = T.matrix('targets')
        symbolic_bias = T.matrix('bias')

        symbolic_weights = T.tensor3('weights')

        prediction = T.nnet.softmax(symbolic_bias+T.batched_dot(symbolic_features, symbolic_weights))
        objective = T.nnet.categorical_crossentropy(prediction, symbolic_targets)

        def onehot(k,K):
            result = np.zeros(K)
            result[k] = 1.0
            return result

#for testing only
        M = 100
        K = self.K
        N = self.N

        test_features = np.random.randn(M,N)
        test_targets = np.array([onehot(np.random.randint(0,K),K) for i in range(M)])
        test_weights = np.random.randn(M,N,K)
        test_bias = np.random.randn(M,K)

        inputs = [normalized_features, [onehot(target, self.K) for target in targets]]
        initial_params = [ np.zeros(self.K), np.random.randn(self.N,self.K) / np.sqrt(self.N) ]

        arsuper(LogisticRegressionProblem, self).__init__(
                symbolic_params = [symbolic_bias, symbolic_weights],
                params= initial_params,
                objective = objective,
                prediction=prediction,
                symbolic_features=[symbolic_features],
                symbolic_targets = symbolic_targets,
                inputs=inputs,
                name=name,
                **kwargs
            )

    def render_params(self):
        return "N={},K={},M={}".format(self.N,self.K,self.M)

#TODO: separate the generic part from the libsvm part
def load_data(filename, mode='cifar'):
    with open(filename, 'r') as f:
        if mode == 'libsvm':

            labels = set()
            all_labels = []
            features = set()
            all_values = []
            for line in f:
                parts = line.split()
                label = int(parts[0])
                all_labels.append(label)
                if label not in labels:
                    labels.add(label)
                values = {}
                for s in parts[1:]:
                    feature_s, value_s = s.split(":")
                    feature = int(feature_s)
                    value = float(value_s)
                    if feature not in features:
                        features.add(feature)
                    values[feature] = value
                all_values.append(values)
            labels = sorted(labels)
            features = sorted(features)
            label_lookup = { label : index for (index, label) in enumerate(labels) }
            feature_lookup = { feature : index for (index, feature) in enumerate(features) }
            n = len(features)
            m = len(all_values)
            inputs = np.zeros((m, n))
            inputs = []
            outputs = []
            for i, label, values in zip(range(m), all_labels, all_values):
                y = label_lookup[label]
                for feature, value in values.items():
                    x[i, feature_lookup[feature]]=value
                outputs.append(y)
            return inputs, outputs

        elif mode == 'kdd':

            inputs = []
            outputs = []
            for line in f:
                entries = line.split()[1:]
                label = int(entries[0])
                #TODO: think about how to handle missing data
                features = [0.0 if x == 'inf' else float(x) for x in entries[1:]]
                inputs.append(features)
                outputs.append(label)
            return inputs, outputs

        elif mode == 'cifar':

            import cPickle
            d = cPickle.load(f)
            return d['data'] / 255.0, d['labels']

        elif mode == 'mnist':

            def parse_int(b):
                a = array.array("i", b)
                a.byteswap()
                return a[0]

            assert filename=="train-images-idx3-ubyte", "Change label file as well."
            label_file ="train-labels-idx1-ubyte"

            f.read(4)
            num_images = parse_int(f.read(4))
            image_height = parse_int(f.read(4))
            image_width = parse_int(f.read(4))
            image_size = image_height * image_width
            images = np.zeros(num_images * image_size)
            data = f.read()
            for i, c in enumerate(data):
                images[i] = ord(c) / 255.0
            images = images.reshape(num_images, image_size)

            with open(label_file, 'r') as g:
                g.read(4)
                num_images = parse_int(g.read(4))
                labels = [ ord(g.read(1)) for i in range(num_images) ]

            return images, labels
