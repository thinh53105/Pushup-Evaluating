import config


class LPF(object):

    def __init__(self, init_value):
        self.init_ang, self.init_fil = init_value
        self.angle_list = [self.init_ang]
        self.filter_list = [self.init_fil]
        self.high = True
        self.count = 0

    def cal_next(self, value):
        self.angle_list.append(value)
        Fn = config.BETA * self.filter_list[-1] + (1 - config.BETA) * value
        self.filter_list.append(Fn)
        state = -1
        if self.high and Fn > value:
            self.count += 0.5
            self.high = False
            state = 1
        if not self.high and Fn < value:
            self.count += 0.5
            self.high = True
            state = 0
        return Fn, state

    def reset(self):
        self.high, self.count = True, 0
        self.angle_list = [self.init_ang]
        self.filter_list = [self.init_fil]