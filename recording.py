import time
from copy import copy
from collections import Counter
from ipdb import set_trace as debug

import matplotlib.pyplot as plt

def get_records(problem=None, optimizer=None, test=None,
        problems=None, optimizers=None,tests=None):

    if problem is not None:
        problems = [problem]
    if optimizer is not None:
        optimizers = [optimizer]

    result = []
    if optimizers is not None:
        for optimizer in optimizers:
            result.extend(optimizer.records)
    elif problems is not None:
        for problem in problems:
            result.extend(problem.records)

    filters = []
    if optimizers is not None:
        filters.append(lambda x : x.optimizer in optimizers)
    if problems is not None:
        filters.append(lambda x : x.problem in problems)
    if test is not None:
        filters.append(test)
    if tests is not None:
        filters.extend(tests)

    for f in filters:
        result = filter(f, result)
    return result

#TODO: print each different problem in a different part of the figure
#(need to remember how to use pyplot better...)
def plot_records(records=None, x='epoch', y='objective', **kwargs):
    if records is None:
        records = get_records(**kwargs)
    for record in records:
        record.plot(x, y)
    plt.legend()
    plt.show()

class Record(object):

    def __init__(self, optimizer, problem=None, note=None):
        if problem is None:
            problem = optimizer.problem
        self.optimizer = optimizer
        self.problem = problem
        problem.records.append(self)
        optimizer.records.append(self)
        self.note = note

        self.costs = Counter()
        self.iteration = 0
        self.epoch_length = problem.epoch_length

        self.snapshots = []
        self.logs = []

        self._logged_time = 0
        self.timing = False
        self.timer_started = None


    def pay_cost(self, cost, name=None):
        self.costs['total'] += cost
        if name is not None:
            self.costs[name] += cost 

    def start_timer(self):
        self.timing = True
        self.timer_started = time.time()

    def stop_timer(self):
        self.timing = False
        self._logged_time += time.time() - self.timer_started
        self.timer_started = None

    @property
    def elapsed_time(self):
        current_timer = time.time() - self.timer_started if self.timing else 0
        return self._logged_time + current_timer

    def plot(self, x, y):
        try:
            ys = [snapshot[y] for snapshot in self.snapshots ]
            xs = [snapshot[x] for snapshot in self.snapshots ]
            line, = plt.plot(xs,ys)
            line.set_label(repr(self.optimizer))
            return line
        except KeyError:
            return None

    def cost(self):
        return self.costs['total']

    def epoch(self):
        return self.cost() * 1.0 / self.epoch_length

    def log(self, entries):
        if entries:
            self.logs.extend([(self.epoch(), self.elapsed_time, self.iteration, entry) for entry in entries])

    def __repr__(self):
        title = "Optimizer: {}\nProblem: {}".format(self.optimizer,self.problem)
        if self.note is not None:
            title = "{} [{}]".format(title, record.note)
        if not self.snapshots:
            return "{}\n[empty record]".format(title)
        else:
            return "{}\nEpochs: {}, Objective: {}".format(
                title, self.snapshots[-1]['epoch'], self.snapshots[-1]['objective']
            )

    def snapshot(self, value=None, display=True, **kwargs):
        snapshot = {}
        if value is not None:
            snapshot['value'] = value
            snapshot['objective'] = self.problem.objective(value, record=self)
        snapshot['costs'] = copy(self.costs)
        snapshot['epoch'] = self.epoch()
        snapshot['iteration'] = self.iteration
        snapshot['elapsed time'] = self.elapsed_time
        snapshot.update(kwargs)
        self.snapshots.append(snapshot)
        if display:
            print(snapshot)
        if value is not None:
            return snapshot['objective']

