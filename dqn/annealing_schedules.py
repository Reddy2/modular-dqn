# Other schedules can be found here: https://www.fys.ku.dk/~andresen/BAhome/ownpapers/permanents/annealSched.pdf

class Constant:
    def __init__(self, value):
        self._value = value

    def value(self, t):
        return self._value


class Linear:
    def __init__(self, start, end, num_steps):
        self._start = float(start)
        self._num_steps = num_steps
        self._interval = (end - start) / (num_steps - 1)

    def value(self, t):
        return self._start + self._interval * min(t, self._num_steps - 1)


class Exponential:
    # T(t) = T(0) * alpha^t, where alpha in (0, 1)
    def __init__(self, start, end, num_steps):
        assert end > 0
        self._start = start
        self._num_steps = num_steps

        # end = start * alpha^(final_t)
        # end/start = alpha^(final_t)
        # ((end/start))^(1/final_t) = alpha
        # Note: final_t = num_steps - 1
        self._alpha = (end / start)**(1 / (num_steps - 1))
        
    def value(self, t):
        return self._start * self._alpha**min(t, self._num_steps - 1)

 
if __name__ == '__main__':
    a = Constant(0.05)
    for i in range(25):
        print(a.value(i))

    print()
    
    b = Linear(1, 0.01, 25)
    for i in range(25):
        print(b.value(i))

    print()

    c = Exponential(1, 0.01, 25)
    for i in range(25):
        print(c.value(i))
