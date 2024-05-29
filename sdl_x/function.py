from .variable import Variable


class Function:
    def __call__(self, input_: Variable) -> Variable:
        self.input_ = input_
        y = self.forward(input_.data)
        self.output = Variable(y)
        self.output.set_creator(self)
        return self.output

    def forward(self, x):
        raise NotImplemented()

    def backward(self, gy):
        raise NotImplemented
