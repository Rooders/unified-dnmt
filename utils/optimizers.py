""" Optimizers class """
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import re
from utils.misc import use_gpu
from torch.cuda.amp import GradScaler

def build_optim(model, opt, checkpoint):
    """ Build optimizer """
    saved_optimizer_state_dict = None

    if opt.train_from and opt.reset_optim != 'all':
        optim = checkpoint['optim']
        # We need to save a copy of optim.optimizer.state_dict() for setting
        # the, optimizer state later on in Stage 2 in this method, since
        # the method optim.set_parameters(model.parameters()) will overwrite
        # optim.optimizer, and with ith the values stored in
        # optim.optimizer.state_dict()
        if opt.reset_optim != 'states':
            saved_optimizer_state_dict = optim.optimizer.state_dict()
            if opt.reset_optim == 'keep_states':
                optim.method = opt.optim
                optim.learning_rate = opt.learning_rate
                optim.original_lr = opt.learning_rate
                optim.max_grad_norm = opt.max_grad_norm
                optim.lr_decay = opt.learning_rate_decay
                optim.start_decay_steps = opt.start_decay_steps
                optim.decay_steps = opt.decay_steps
                optim.betas = [opt.adam_beta1, opt.adam_beta2]
                optim.adagrad_accum = opt.adagrad_accumulator_init
                optim.decay_method = opt.decay_method
                optim.warmup_steps = opt.warmup_steps
                optim.model_size = opt.dec_rnn_size
                optim.mixed_precision = opt.mixed_precision
                optim.doc_double_lr=opt.doc_double_lr
                optim.doc_lr=opt.doc_double_lr
    else:
        optim = Optimizer(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_steps=opt.start_decay_steps,
            decay_steps=opt.decay_steps,
            beta1=opt.adam_beta1,
            beta2=opt.adam_beta2,
            adagrad_accum=opt.adagrad_accumulator_init,
            decay_method=opt.decay_method,
            warmup_steps=opt.warmup_steps,
            model_size=opt.dec_rnn_size,
            mixed_precision=opt.mixed_precision,
            doc_double_lr=opt.doc_double_lr, doc_lr=opt.doc_double_lr
            )

    # Stage 1:
    # Essentially optim.set_parameters (re-)creates and optimizer using
    # model.paramters() as parameters that will be stored in the
    # optim.optimizer.param_groups field of the torch optimizer class.
    # Importantly, this method does not yet load the optimizer state, as
    # essentially it builds a new optimizer with empty optimizer state and
    # parameters from the model.
    
    
    if opt.train_from and opt.doc_double_lr:
      doc_params = []
      sent_params = []
      def fix_key(s):
        s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
                    r'\1.layer_norm\2.bias', s)
        s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
                    r'\1.layer_norm\2.weight', s)
        return s

      checkpoint['model'] = \
        {fix_key(k): v for (k, v) in checkpoint['model'].items()}
      
      for (k, v) in model.named_parameters():
        if v.requires_grad:
          if k not in checkpoint['model'].keys():
            doc_params.append(v)
          else:
            sent_params.append(v)
      param_g = [{'params':sent_params}, {'params':doc_params}]   
    else:
      param_g = []
    
    
    # checkpoint['model'] = \
    #   {fix_key(k): v for (k, v) in checkpoint['model'].items()}
    optim.set_parameters(model.named_parameters(), param_g)

    if opt.train_from and (opt.reset_optim in ['none', 'keep_states']):
        # Stage 2: In this stage, which is only performed when loading an
        # optimizer from a checkpoint, we load the saved_optimizer_state_dict
        # into the re-created optimizer, to set the optim.optimizer.state
        # field, which was previously empty. For this, we use the optimizer
        # state saved in the "saved_optimizer_state_dict" variable for
        # this purpose.
        # See also: https://github.com/pytorch/pytorch/issues/2830
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        # Convert back the state values to cuda type if applicable
        if use_gpu(opt):
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        # We want to make sure that indeed we have a non-empty optimizer state
        # when we loaded an existing model. This should be at least the case
        # for Adam, which saves "exp_avg" and "exp_avg_sq" state
        # (Exponential moving average of gradient and squared gradient values)
        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    return optim


class MultipleOptimizer(object):
    """ Implement multiple optimizers needed for sparse adam """

    def __init__(self, op):
        """ ? """
        self.optimizers = op

    def zero_grad(self):
        """ ? """
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        """ ? """
        for op in self.optimizers:
            op.step()

    @property
    def state(self):
        """ ? """
        return {k: v for op in self.optimizers for k, v in op.state.items()}

    def state_dict(self):
        """ ? """
        return [op.state_dict() for op in self.optimizers]

    def load_state_dict(self, state_dicts):
        """ ? """
        assert len(state_dicts) == len(self.optimizers)
        for i in range(len(state_dicts)):
            self.optimizers[i].load_state_dict(state_dicts[i])


class Optimizer(object):
    """
    Controller class for optimization. Mostly a thin
    wrapper for `optim`, but also useful for implementing
    rate scheduling beyond what is currently available.
    Also implements necessary methods for training RNNs such
    as grad manipulations.

    Args:
      method (:obj:`str`): one of [sgd, adagrad, adadelta, adam]
      lr (float): learning rate
      lr_decay (float, optional): learning rate decay multiplier
      start_decay_steps (int, optional): step to start learning rate decay
      beta1, beta2 (float, optional): parameters for adam
      adagrad_accum (float, optional): initialization parameter for adagrad
      decay_method (str, option): custom decay options
      warmup_steps (int, option): parameter for `noam` decay
      model_size (int, option): parameter for `noam` decay

    We use the default parameters for Adam that are suggested by
    the original paper https://arxiv.org/pdf/1412.6980.pdf
    These values are also used by other established implementations,
    e.g. https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
    https://keras.io/optimizers/
    Recently there are slightly different values used in the paper
    "Attention is all you need"
    https://arxiv.org/pdf/1706.03762.pdf, particularly the value beta2=0.98
    was used there however, beta2=0.999 is still arguably the more
    established value, so we use that here as well
    """

    def __init__(self, method, learning_rate, max_grad_norm,
                 lr_decay=1, start_decay_steps=None, decay_steps=None,
                 beta1=0.9, beta2=0.999,
                 adagrad_accum=0.0,
                 decay_method=None,
                 warmup_steps=4000,
                 model_size=None, mixed_precision=False, doc_double_lr=0, doc_lr=1.0):
        self.last_ppl = None
        self.learning_rate = learning_rate
        self.original_lr = learning_rate
        self.max_grad_norm = max_grad_norm
        self.method = method
        
        self.lr_decay = lr_decay
        self.start_decay_steps = start_decay_steps
        self.decay_steps = decay_steps
        self.start_decay = False
        
        self._step = 0
        self.betas = [beta1, beta2]
        self.adagrad_accum = adagrad_accum
        self.decay_method = decay_method
        
        self.warmup_steps = warmup_steps
        self.model_size = model_size
        self.mixed_precision = mixed_precision
        
        self.doc_double_lr = doc_double_lr
        self.doc_lr = doc_lr
        # for inverse_sqrt LR
        if self.decay_method == "inverse_sqrt":
          self.learn_rate = 0.0
        self.lr_step = self.original_lr / self.warmup_steps 
        self.decay_factor = self.original_lr * self.warmup_steps ** 0.5

    def set_parameters(self, params, params_g=[]):
        """ ? """
        self.params = []
        self.sparse_params = []
        if self.mixed_precision:
           self.scaler = GradScaler()
        
        if params_g:
           self.params = params_g
           self.params[0]['lr'] = self.learning_rate * self.doc_lr
           self.params[1]['lr'] = self.learning_rate
        else:
           for k, p in params:
               if p.requires_grad:
                   if self.method != 'sparseadam' or "embed" not in k:
                      self.params.append(p)
                   else:
                      self.sparse_params.append(p)
        
        # for k, p in params:
        #        if p.requires_grad:
        #            if self.method != 'sparseadam' or "embed" not in k:
        #               self.params.append(p)
        #            else:
        #               self.sparse_params.append(p)


        # Noth: the double lr just apply for adam optimizer 
        
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.learning_rate)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.learning_rate)
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    self.optimizer.state[p]['sum'] = self.optimizer\
                        .state[p]['sum'].fill_(self.adagrad_accum)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.learning_rate)
        elif self.method == 'adam':
            if params_g:
              self.optimizer = optim.Adam(self.params,
                                        betas=self.betas, eps=1e-9) 
            else:
              self.optimizer = optim.Adam(self.params, lr=self.learning_rate,
                                        betas=self.betas, eps=1e-9)
            # self.optimizer = optim.Adam(self.params, lr=self.learning_rate,
            #                             betas=self.betas, eps=1e-9)
        
        elif self.method == 'sparseadam':
            self.optimizer = MultipleOptimizer(
                [optim.Adam(self.params, lr=self.learning_rate,
                            betas=self.betas, eps=1e-8),
                 optim.SparseAdam(self.sparse_params, lr=self.learning_rate,
                                  betas=self.betas, eps=1e-8)])
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def _set_rate(self, learning_rate):
        self.learning_rate = learning_rate
        if self.method != 'sparseadam':
            self.optimizer.param_groups[0]['lr'] = self.learning_rate
            if self.doc_double_lr:
              self.optimizer.param_groups[0]['lr'] = self.learning_rate * self.doc_lr
              self.optimizer.param_groups[1]['lr'] = self.learning_rate 
        else:
            for op in self.optimizer.optimizers:
                op.param_groups[0]['lr'] = self.learning_rate
                if self.doc_double_lr:
                  op.param_groups[0]['lr'] = self.learning_rate * self.doc_lr
                  op.param_groups[1]['lr'] = self.learning_rate
    
    def update_lr(self):
        # Decay method used in tensor2tensor.
        if self.decay_method == "noam":
            self.learn_rate = self.original_lr * (self.model_size ** (-0.5) * min(self._step ** (-0.5), self._step * self.warmup_steps**(-1.5)))
        
        # Decay method in fairseq
        elif self.decay_method == "inverse_sqrt":
          if self._step <= self.warmup_steps:
            self.learn_rate = self.lr_step + self.learn_rate
          else:
            self.learn_rate = self.decay_factor * self._step**(-0.5)
        
        # Decay based on start_decay_steps every decay_steps
        else:
            if ((self.start_decay_steps is not None) and (
                     self._step >= self.start_decay_steps)):
                self.start_decay = True
            if self.start_decay:
                if ((self._step - self.start_decay_steps)
                   % self.decay_steps == 0):
                    self.learning_rate = self.learning_rate * self.lr_decay

    def step(self):
        """Update the model parameters based on current gradients.

        Optionally, will employ gradient modification or update learning
        rate.
        """
        self._step += 1

        
        self.update_lr()
        self._set_rate(self.learn_rate)
        # if self.method != 'sparseadam':
        #     self._
        #     self.optimizer.param_groups[0]['lr'] = self.learning_rate
        #     if self.doc_double_lr:
        #       self.optimizer.param_groups[1]['lr'] = self.learning_rate * self.doc_lr
        
        if self.mixed_precision and self.max_grad_norm > 0:
          self.scaler.unscale_(self.optimizer)
        
        if self.max_grad_norm:
          clip_grad_norm_(self.params, self.max_grad_norm)
        
        if self.mixed_precision:
          self.scaler.step(self.optimizer)
          self.scaler.update()
        
        else:
          self.optimizer.step()
