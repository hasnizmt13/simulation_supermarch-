import simpy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Supermarket:
    def __init__(self, env, num_cashiers=4, extra_cashier_policy=4, lambda_rate=4, C=3):
        self.env = env
        self.num_cashiers = num_cashiers
        self.extra_cashier_policy = extra_cashier_policy
        self.lambda_rate = lambda_rate
        self.C = C
        self.cashiers = [simpy.Resource(env) for _ in range(num_cashiers)]
        self.lost_customers = 0
        self.profit = 0
        self.waiting_times = []
        self.open_extra_cashier = False
        self.extra_cashier = simpy.Resource(env)

    def customer_arrival(self):
        while True:
            arrival_interval = np.random.exponential(1.0 / self.lambda_rate)
            yield self.env.timeout(arrival_interval)
            arrival_time = self.env.now
            self.env.process(self.customer_process(arrival_time))

    def customer_process(self, arrival_time):
        cashier = self.choose_cashier()
        if cashier is not None:
            with cashier.request() as request:
                result = yield request | self.env.timeout(1)  # Clients leave after waiting 1 unit of time
                if request in result:
                    service_start_time = self.env.now
                    self.waiting_times.append(service_start_time - arrival_time)
                    service_time = np.random.exponential(1.0)
                    yield self.env.timeout(service_time)
                    self.profit += 10  # Gain from servicing a customer
                else:
                    self.lost_customers += 1  # Increment lost customers
        else:
            self.lost_customers += 1  # No cashier was available, and no extra cashier opened

    def choose_cashier(self):
        queues = [(len(c.queue), c) for c in self.cashiers]
        queues.sort(key=lambda x: x[0])
        min_queue, min_cashier = queues[0]

        if self.open_extra_cashier:
            return self.extra_cashier

        if not self.open_extra_cashier and min_queue >= self.extra_cashier_policy:
            self.env.process(self.open_extra_cashier_process())

        return min_cashier if min_queue < 5 else None  # Return None if all queues are full and no extra cashier

    def open_extra_cashier_process(self):
        if not self.open_extra_cashier:
            self.open_extra_cashier = True
            yield self.env.timeout(2)  # Delay to open the extra cashier
            self.open_extra_cashier = False
            self.profit -= 2 * self.C  # Cost for opening the cashier

def simulate(supermarket_params, num_runs=30):
    profits, lost_customers, avg_waiting_times = [], [], []
    for _ in range(num_runs):
        env = simpy.Environment()
        supermarket = Supermarket(env, **supermarket_params)
        env.process(supermarket.customer_arrival())
        env.run(until=200)
        profits.append(supermarket.profit)
        avg_waiting_times.append(np.mean(supermarket.waiting_times))
        lost_customers.append(supermarket.lost_customers)

    mean_profit = np.mean(profits)
    ci_profit = 1.96 * np.std(profits) / np.sqrt(num_runs)
    mean_waiting_time = np.mean(avg_waiting_times)
    ci_waiting_time = 1.96 * np.std(avg_waiting_times) / np.sqrt(num_runs)

    return (mean_profit, ci_profit), (mean_waiting_time, ci_waiting_time)

# Parameters and simulation
policies = [0, 2, 4, 6]
lambda_rates = np.arange(1, 10, 1)
results = {}

# Valeurs de C à tester
C_values = [1, 3, 5]  # Par exemple, tester C = 1, C = 3, et C = 5

# Paramètres et simulations pour chaque valeur de C
for C in C_values:
    results = {}
    for policy in policies:
        for lambda_rate in lambda_rates:
            params = {'extra_cashier_policy': policy, 'lambda_rate': lambda_rate, 'C': C}
            results[(policy, lambda_rate)] = simulate(params)

    # Tracé des résultats pour la valeur actuelle de C
    plt.figure(figsize=(12, 8))
    for policy in policies:
        profits = [results[(policy, lr)][0][0] for lr in lambda_rates]
        errors = [results[(policy, lr)][0][1] for lr in lambda_rates]
        plt.errorbar(lambda_rates, profits, yerr=errors, label=f'Policy {policy}')
    plt.xlabel(f'Lambda Rate for C = {C}')
    plt.ylabel('Mean Profit')
    plt.title(f'Mean Profits by Policy with 95% CI for C = {C}')
    plt.legend()
    plt.show()

