import flwr as fl
from flwr.server.strategy import FedAvg

config = {
    'num_clients': 2,
    'batch_size': 4,
    'num_rounds': 5,
    'epochs_per_round': 1,
    'validation_split': .2,
    'seed': 41
}

def start_server(num_rounds, num_clients, fraction_fit=1.0):
    strategy = FedAvg(min_available_clients=num_clients,
                      fraction_fit=fraction_fit)
    # Exposes the server by default on port 8080
    fl.server.start_server(strategy=strategy, config={
                           "num_rounds": num_rounds})


if __name__=='__main__':
    start_server(num_rounds=config['num_rounds'], num_clients=config['num_clients'])
