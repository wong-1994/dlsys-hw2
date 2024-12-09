"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
            if p.grad is None:
                continue
            if p not in self.u:
                self.u[p] = 0
            u = self.momentum * self.u[p] + (1 - self.momentum) * (p.grad.data + self.weight_decay * p.data)
            self.u[p] = ndl.Tensor(u.data, dtype=p.dtype)
            p.data -= self.lr * self.u[p].data
        return
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            if p not in self.m:
                self.m[p] = 0
            if p not in self.v:
                self.v[p] = 0
            grad = p.grad.data + self.weight_decay * p.data
            m = self.beta1 * self.m[p] + (1 - self.beta1) * grad
            v = self.beta2 * self.v[p] + (1 - self.beta2) * (grad ** 2)
            self.m[p] = ndl.Tensor(m.data, dtype=p.dtype)
            self.v[p] = ndl.Tensor(v.data, dtype=p.dtype)
            m = m / (1 - self.beta1 ** self.t)
            v = v / (1 - self.beta2 ** self.t)
            p.data -= ndl.Tensor(self.lr * m / (v ** 0.5 + self.eps), dtype=p.dtype)
        return
        ### END YOUR SOLUTION
