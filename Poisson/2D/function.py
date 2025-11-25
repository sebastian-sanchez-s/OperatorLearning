class Function:
    def __init__(self, f1: callable = None):
        self.expr = []
        if f1: self.expr.append(f1)

    def copy(self):
        f = Function()
        f.expr = self.expr.copy()
        return f

    def __call__(self, x, *args, **kwargs):
        ans = self.expr[0](x)
        other = None
        for e in self.expr[1:]:
            if callable(e):
                other = e(x)
            elif not isinstance(e, str):
                other = e
            else:
                match e:
                    case '+': ans = ans + other
                    case '-': ans = ans - other
                    case '*': ans = ans * other
                    case '/': ans = ans / other
                    case '//': ans = ans // other
        return ans

    def __add_expr(self, other, expr: str):
        fun1 = Function()
        fun1.expr = self.expr.copy()

        if callable(other):
            fun2 = Function()
            fun2.expr = other.expr.copy()
            fun1.expr.append(fun2)
        elif isinstance(other, (int, float, complex)):
            fun1.expr.append(other)
        else:
            fun1.expr.append(other)

        fun1.expr.append(expr)

        return fun1

    def __add__(self, other): return self.__add_expr(other, '+')
    def __sub__(self, other): return self.__add_expr(other, '-')
    def __mul__(self, other): return self.__add_expr(other, '*')
    def __truediv__(self, other): return self.__add_expr(other, '/')
    def __div__(self, other): return self.__add_expr(other, '//')