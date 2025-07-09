import neat
import loader
import pickle
import pandas as pd
import argparse
import os
import visualize  


def compute_mse(net, X, Y):
    return sum((Y[i] - net.activate(X[i])[0])**2 for i in range(len(Y))) / len(Y)


def make_eval_genomes(trainX, trainY, csv_path):
    best_fitness_per_generation = []

    def eval_genomes(genomes, config):
        nonlocal best_fitness_per_generation
        max_fitness = None

        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            cost = sum((net.activate(trainX[i])[0] - trainY[i]) ** 2 for i in range(len(trainY)))
            genome.fitness = float(-cost / len(trainY))

            if max_fitness is None or genome.fitness > max_fitness:
                max_fitness = genome.fitness

        best_fitness_per_generation.append(max_fitness)

        df = pd.DataFrame({
            'Generation': range(1, len(best_fitness_per_generation) + 1),
            'Best Fitness': best_fitness_per_generation
        })
        df.to_csv(csv_path, index=False)

    return eval_genomes


def run_eval(config_path, num_generations, pop_size, net_name):
    # Setup output directory
    output_dir = os.path.join("runs", net_name)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "fitness.csv")
    pkl_path = os.path.join(output_dir, "best_network.pkl")

    # Load config
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    config.pop_size = pop_size

    # Load training and testing data
    trainX, trainY, testX, testY = loader.load_data(split=0.8)

    # Create population
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run NEAT
    eval_fn = make_eval_genomes(trainX, trainY, csv_path)
    winner = p.run(eval_fn, num_generations)

    # Save best genome
    with open(pkl_path, "wb") as f:
        pickle.dump(winner, f)

    # Final trained network
    best_net = neat.nn.FeedForwardNetwork.create(winner, config)
    train_cost = compute_mse(best_net, trainX, trainY)
    test_cost = compute_mse(best_net, testX, testY)

    print(f"\nTrain MSE: {train_cost}")
    print(f"Test MSE: {test_cost}")

    # Visualization
    net_path = os.path.join(output_dir, "network")
    fitness_plot = os.path.join(output_dir, "fitness_history.png")
    species_plot = os.path.join(output_dir, "species_evolution.png")

    node_names = None  
    visualize.draw_net(config, winner, True, node_names=node_names, filename=net_path)
    visualize.plot_stats(stats, ylog=False, view=True, filename=fitness_plot)
    visualize.plot_species(stats, view=True, filename=species_plot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train NEAT with visualization and flexible setup.")
    parser.add_argument('--gens', type=int, default=20, help="Number of generations")
    parser.add_argument('--pop_size', type=int, default=20, help="Population size")
    parser.add_argument('--net_name', type=str, default="default_run", help="Name of your network")
    parser.add_argument('--config', type=str, default="config.txt", help="Path to NEAT config file")
    args = parser.parse_args()

    run_eval(args.config, args.gens, args.pop_size, args.net_name)
