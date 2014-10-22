from layers import Linear, ReLU, dropout, fast_dropout
from classifiers import LogisticRegression
from collections import OrderedDict
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import shared
import sys


def build_shared_zeros(shape, name):
    return shared(value=numpy.zeros(shape, dtype=theano.config.floatX),
            name=name, borrow=True)


class CrossNet(object):
    def __init__(self, numpy_rng, theano_rng=None, 
            n_ins=(4096, 40*50),
            layers_types=([ReLU], [ReLU, ReLU, ReLU]),
            layers_sizes=([], [1024, 1024]),
            n_outs=200,
            loss='cos_cos2',
            rho=0.95, eps=1.E-6,
            l1_reg=0.,
            l2_reg=0.,
            max_norm=0.,
            debugprint=False):
        self.layers = []
        self.params = []
        self.n_layers = len(layers_types)
        self.layers_types = layers_types
        assert self.n_layers > 0
        self.max_norm = max_norm
        self._rho = rho  # ``momentum'' for adadelta
        self._eps = eps  # epsilon for adadelta
        self._accugrads = []  # for adadelta
        self._accudeltas = []  # for adadelta

        if theano_rng == None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.img = T.fmatrix('img')
        self.snd = T.fmatrix('snd')
        self.y = T.ivector('y')
        
        self.layers_ins_img = [n_ins[0]] + layers_sizes[0]
        self.layers_ins_snd = [n_ins[1]] + layers_sizes[1]
        self.layers_outs_img = layers_sizes[0] + [n_outs]
        self.layers_outs_snd = layers_sizes[1] + [n_outs]
        layer_input_img = self.img
        layer_input_snd = self.snd

        for layer_type, n_in, n_out in zip(layers_types[0],
                self.layers_ins_img, self.layers_outs_img):
            this_layer_img = layer_type(rng=numpy_rng,
                    #input=layer_input_img, n_in=n_in, n_out=n_out)
                    input=layer_input_img, n_in=n_in, n_out=n_out, cap=6.)
            assert hasattr(this_layer_img, 'output')
            layer_input_img = this_layer_img.output
            self.params.extend(this_layer_img.params)
            self._accugrads.extend([build_shared_zeros(t.shape.eval(),
                'accugrad') for t in this_layer_img.params])
            self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
                'accudelta') for t in this_layer_img.params])
            self.layers.append(this_layer_img)

        for layer_type, n_in, n_out in zip(layers_types[1],
                self.layers_ins_snd, self.layers_outs_snd):
            this_layer_snd = layer_type(rng=numpy_rng,
                    #input=layer_input_snd, n_in=n_in, n_out=n_out)
                    input=layer_input_snd, n_in=n_in, n_out=n_out, cap=6.)
            assert hasattr(this_layer_snd, 'output')
            layer_input_snd = this_layer_snd.output
            self.params.extend(this_layer_snd.params)
            self._accugrads.extend([build_shared_zeros(t.shape.eval(),
                'accugrad') for t in this_layer_snd.params])
            self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
                'accudelta') for t in this_layer_snd.params])
            self.layers.append(this_layer_snd)

        L2 = 0.
        for param in self.params:
            L2 += T.sum(param ** 2)
        L1 = 0.
        for param in self.params:
            L1 += T.sum(abs(param))

        self.squared_error = (layer_input_img - layer_input_snd).norm(2, axis=-1) **2
        self.mse = T.mean(self.squared_error, axis=-1)
        self.rmse = T.sqrt(self.mse)
        self.sse = T.sum(self.squared_error, axis=-1)
        self.rsse = T.sqrt(self.sse)

        self.rsse_cost = T.switch(self.y, self.rsse, -self.rsse)
        self.rmse_cost = T.switch(self.y, self.rmse, -self.rmse)
        self.sum_rmse_costs = T.sum(self.rmse_cost)
        self.sum_rsse_costs = T.sum(self.rsse_cost)
        self.mean_rmse_costs = T.mean(self.rmse_cost)
        self.mean_rsse_costs = T.mean(self.rsse_cost)

	self.dot_prod = T.sum(layer_input_img * layer_input_snd, axis=-1)
        #self.cos_sim = (T.batched_dot(layer_input_img, layer_input_snd) / 
        self.cos_sim = (self.dot_prod /
            (layer_input_img.norm(2, axis=-1) * layer_input_snd.norm(2, axis=-1)))
        self.cos_sim_cost = T.switch(self.y, 1.-self.cos_sim, self.cos_sim)
        self.mean_cos_sim_cost = T.mean(self.cos_sim_cost)
        self.sum_cos_sim_cost = T.sum(self.cos_sim_cost)

        #self.cos_sim_cost = T.switch(self.y, 1.-abs(self.cos_sim), abs(self.cos_sim))
        self.cos2_sim_cost = T.switch(self.y, 1.-(self.cos_sim ** 2), self.cos_sim ** 2)
        self.mean_cos2_sim_cost = T.mean(self.cos2_sim_cost)
        self.sum_cos2_sim_cost = T.sum(self.cos2_sim_cost)

        self.cos_cos2_sim_cost = T.switch(self.y, (1.-self.cos_sim)/2, self.cos_sim ** 2)
        self.mean_cos_cos2_sim_cost = T.mean(self.cos_cos2_sim_cost)
        self.sum_cos_cos2_sim_cost = T.sum(self.cos_cos2_sim_cost)

        from layers import relu_f
        #self.margin_dot_cost = relu_f(T.switch(self.y, 1.-self.dot_prod, -1.+self.dot_prod))
        self.margin_dot_cost = T.switch(self.y, 1.-self.dot_prod, -1.+self.dot_prod)
	self.mean_margin_dot_cost = T.mean(self.margin_dot_cost)
	self.sum_margin_dot_cost = T.sum(self.margin_dot_cost)

        self.euclidean = (layer_input_img - layer_input_snd).norm(2, axis=-1)
        self.euclidean_cost = T.switch(self.y, self.euclidean, -self.euclidean)
        self.mean_euclidean_cost = T.mean(self.euclidean_cost)
        self.sum_euclidean_cost = T.sum(self.euclidean_cost)

        self.normalized_euclidean = ((layer_input_img - layer_input_snd).norm(2, axis=-1) / (layer_input_img.norm(2, axis=-1) * layer_input_snd.norm(2, axis=-1)))
        self.normalized_euclidean_cost = T.switch(self.y, self.normalized_euclidean, -self.normalized_euclidean)
        self.mean_normalized_euclidean_cost = T.mean(self.normalized_euclidean_cost)
        self.sum_normalized_euclidean_cost = T.sum(self.normalized_euclidean_cost)

        self.hellinger = 0.5 * T.sqrt(T.sum((T.sqrt(layer_input_img) - T.sqrt(layer_input_snd))**2, axis=1))
        self.hellinger_cost = T.switch(self.y, self.hellinger, 1.-self.hellinger)
        self.mean_hellinger_cost = T.mean(self.hellinger_cost)
        self.sum_hellinger_cost = T.sum(self.hellinger_cost)

        self.layer_output_img = layer_input_img
        self.layer_output_snd = layer_input_snd

        if loss == 'cos_cos2':
            self.cost = self.sum_cos_cos2_sim_cost
            self.mean_cost = self.mean_cos_cos2_sim_cost
        elif loss == 'cos':
            self.cost = self.sum_cos_sim_cost
            self.mean_cost = self.mean_cos_sim_cost
        elif loss == 'cos2':
            self.cost = self.sum_cos2_sim_cost
            self.mean_cost = self.mean_cos_sim_cost
        elif loss == 'margin_dot':
	    self.cost = self.sum_margin_dot_cost
            self.mean_cost = self.mean_margin_dot_cost
        elif loss == 'euclidean':
            self.cost = self.sum_euclidean_cost
            self.mean_cost = self.mean_euclidean_cost
        elif loss == 'norm_euclidean':
            self.cost = self.sum_normalized_euclidean_cost
            self.mean_cost = self.mean_normalized_euclidean_cost
        elif loss == 'hellinger':
            self.cost = self.sum_hellinger_cost
            self.mean_cost = self.mean_hellinger_cost
        else:
            print >> sys.stderr, "NO COST FUNCTION"
            sys.exit(-1)

        if l1_reg:
            self.cost = self.cost + l1_reg*L1
            self.mean_cost = self.mean_cost + l1_reg*L1
        if l2_reg:
            self.cost = self.cost + l2_reg*L2
            self.mean_cost = self.mean_cost + l2_reg*L2

        if debugprint:
            theano.printing.debugprint(self.cost)

        if hasattr(self, 'cost'):
            self.cost_training = self.cost
        if hasattr(self, 'mean_cost'):
            self.mean_cost_training = self.mean_cost

    def __repr__(self):
        dimensions_layers_str = (map(lambda x: "x".join(map(str, x)),
                zip(self.layers_ins_img, self.layers_outs_img)),
                map(lambda x: "x".join(map(str, x)),
                zip(self.layers_ins_snd, self.layers_outs_snd)))

        return ", ".join(["_".join(map(lambda x: "_".join((x[0].__name__, x[1])),
            zip(self.layers_types[0], dimensions_layers_str[0]))),
            "_".join(map(lambda x: "_".join((x[0].__name__, x[1])),
            zip(self.layers_types[1], dimensions_layers_str[1])))])

    def get_SGD_trainer(self, debug=False):
        """ Returns a plain SGD minibatch trainer with learning rate as param.
        """
        batch_img = T.fmatrix('batch_img')
        batch_snd = T.fmatrix('batch_snd')
        batch_y = T.ivector('batch_y')
        learning_rate = T.fscalar('lr')  # learning rate to use
        # compute the gradients with respect to the model parameters
        # using mean_cost so that the learning rate is not too dependent on the batch size
        cost = self.mean_cost_training
        gparams = T.grad(cost, self.params)

        # compute list of weights updates
        updates = OrderedDict()
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - gparam * learning_rate 

        outputs = cost
        if debug:
            outputs = [cost] + self.params + gparams +\
                    [updates[param] for param in self.params]

        train_fn = theano.function(inputs=[theano.Param(batch_img), 
            theano.Param(batch_snd), theano.Param(batch_y),
            theano.Param(learning_rate)],
            outputs=outputs,
            updates=updates,
            givens={self.img: batch_img, self.snd: batch_snd, self.y: batch_y})

        return train_fn

    def get_adadelta_trainer(self, debug=False):
        batch_img = T.fmatrix('batch_img')
        batch_snd = T.fmatrix('batch_snd')
        batch_y = T.ivector('batch_y')
        # compute the gradients with respect to the model parameters
        cost = self.cost_training
        gparams = T.grad(cost, self.params)

        # compute list of weights updates
        updates = OrderedDict()
        for accugrad, accudelta, param, gparam in zip(self._accugrads,
                self._accudeltas, self.params, gparams):
            # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
            agrad = self._rho * accugrad + (1 - self._rho) * gparam * gparam
            dx = - T.sqrt((accudelta + self._eps) / (agrad + self._eps)) * gparam
            updates[accudelta] = self._rho * accudelta + (1 - self._rho) * dx * dx
            if self.max_norm:
                W = param + dx
                col_norms = W.norm(2, axis=0)
                desired_norms = T.clip(col_norms, 0, self.max_norm)
                updates[param] = W * (desired_norms / (1e-6 + col_norms))
            else:
                updates[param] = param + dx
            updates[accugrad] = agrad

        outputs = cost
        if debug:
            outputs = [cost] + self.params + gparams +\
                    [updates[param] for param in self.params]# +\

        train_fn = theano.function(inputs=[theano.Param(batch_img), 
            theano.Param(batch_snd), theano.Param(batch_y)],
            outputs=outputs,
            updates=updates,
            givens={self.img: batch_img, self.snd: batch_snd, self.y: batch_y})

        return train_fn

    def score_classif(self, given_set):
        batch_img = T.fmatrix('batch_img')
        batch_snd = T.fmatrix('batch_snd')
        batch_y = T.ivector('batch_y')
        score = theano.function(inputs=[theano.Param(batch_img), 
            theano.Param(batch_snd), theano.Param(batch_y)],
                outputs=self.mean_cost,
                givens={self.img: batch_img, self.snd: batch_snd, self.y: batch_y})

        # Create a function that scans the entire set given as input
        def scoref():
            return [score(img, snd, y) for (img, snd, y) in given_set]

        return scoref

    def score_classif_same_diff_separated(self, given_set):
        batch_img = T.fmatrix('batch_img')
        batch_snd = T.fmatrix('batch_snd')
        batch_y = T.ivector('batch_y')
        #cost_same = T.mean(self.cos_sim[T.eq(self.y, 1).nonzero()], axis=-1)
        #cost_diff = T.mean(self.cos_sim[T.eq(self.y, 0).nonzero()], axis=-1)
        cost_same = T.mean(self.dot_prod[T.eq(self.y, 1).nonzero()], axis=-1)
        cost_diff = T.mean(self.dot_prod[T.eq(self.y, 0).nonzero()], axis=-1)
        score = theano.function(inputs=[theano.Param(batch_img), 
            theano.Param(batch_snd), theano.Param(batch_y)],
                outputs=[cost_same, cost_diff],
                givens={self.img: batch_img, self.snd: batch_snd, self.y: batch_y})
        # Create a function that scans the entire set given as input
        def scoref():
            return [score(img, snd, y) for (img, snd, y) in given_set]

        return scoref

    def transform_img_snd(self):
        batch_img = T.fmatrix('batch_img')
        batch_snd = T.fmatrix('batch_snd')
        transform = theano.function(inputs=[theano.Param(batch_img), 
            theano.Param(batch_snd)],
                #outputs=[self.layer_output_img, self.layer_output_snd],
                outputs=[self.layers[len(self.layers_outs_img) - 1].output,
                    self.layers[len(self.layers_outs_img) + len(self.layers_outs_snd) - 1].output],
                givens={self.img: batch_img, self.snd: batch_snd})
        return transform

    def transform_img(self):
        batch_img = T.fmatrix('batch_img')
        transform = theano.function(inputs=[theano.Param(batch_img)],
                #outputs=self.layer_output_img,
                outputs=self.layers[len(self.layers_outs_img) - 1].output, 
                givens={self.img: batch_img})
        return transform

    def transform_snd(self):
        batch_snd = T.fmatrix('batch_snd')
        transform = theano.function(inputs=[theano.Param(batch_snd)],
                #outputs=self.layer_output_snd,
                outputs=self.layers[len(self.layers_outs_img) + len(self.layers_outs_snd) - 1].output,
                givens={self.snd: batch_snd})
        return transform


class DropoutCrossNet(CrossNet):
    pass # TODO
