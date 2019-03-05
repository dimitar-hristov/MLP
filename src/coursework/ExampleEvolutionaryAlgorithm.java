package coursework;

import java.util.ArrayList;

import model.Fitness;
import model.Individual;
import model.LunarParameters.DataSet;
import model.NeuralNetwork;

/**
 * Implements a basic Evolutionary Algorithm to train a Neural Network
 * 
 * You Can Use This Class to implement your EA or implement your own class that extends {@link NeuralNetwork} 
 * 
 */
public class ExampleEvolutionaryAlgorithm extends NeuralNetwork {
	

	/**
	 * The Main Evolutionary Loop
	 */
	@Override
	public void run() {		
		//Initialise a population of Individuals with random weights
		population = initialise();

		//Record a copy of the best Individual in the population
		best = getBest();
		System.out.println("Best From Initialisation " + best);

		/**
		 * main EA processing loop
		 */		
		
		while (evaluations < Parameters.maxEvaluations) {

			/**
			 * this is a skeleton EA - you need to add the methods.
			 * You can also change the EA if you want 
			 * You must set the best Individual at the end of a run
			 * 
			 */

			// Select 2 Individuals from the current population. Currently returns random Individual
			Individual parent1 = select(); 
			Individual parent2 = select();

			// Generate a child by crossover. Not Implemented			
			ArrayList<Individual> children = reproduce(parent1, parent2);			
			
			//mutate the offspring
			mutate(children);
			
			// Evaluate the children
			evaluateIndividuals(children);			

			// Replace children in population
			replace(children);

			// check to see if the best has improved
			best = getBest();
			
			// Implemented in NN class. 
			outputStats();
			
			//Increment number of completed generations			
		}

		//save the trained network to disk
		saveNeuralNetwork();
	}

	

	/**
	 * Sets the fitness of the individuals passed as parameters (whole population)
	 * 
	 */
	private void evaluateIndividuals(ArrayList<Individual> individuals) {
		for (Individual individual : individuals) {
			individual.fitness = Fitness.evaluate(individual, this);
		}
	}


	/**
	 * Returns a copy of the best individual in the population
	 * 
	 */
	private Individual getBest() {
		best = null;;
		for (Individual individual : population) {
			if (best == null) {
				best = individual.copy();
			} else if (individual.fitness < best.fitness) {
				best = individual.copy();
			}
		}
		return best;
	}

	/**
	 * Generates a randomly initialised population
	 * 
	 */
	private ArrayList<Individual> initialise() {
		population = new ArrayList<>();
		for (int i = 0; i < Parameters.popSize; ++i) {
			//chromosome weights are initialised randomly in the constructor
			Individual individual = new Individual();
			population.add(individual);
		}
		evaluateIndividuals(population);
		return population;
	}

	/**
	 * Selection --
	 * 
	 * NEEDS REPLACED with proper selection this just returns a copy of a random
	 * member of the population
	 */
	private Individual select() {		
		//Individual parent = population.get(Parameters.random.nextInt(Parameters.popSize));
		//return parent.copy();
		
		/*
		 * ROULETTE
		 */
		double fitnessSum = 0;
		for (int i = 0; i < population.size(); i++) {
			fitnessSum += 1.0 / population.get(i).fitness;
		}
		
		Individual parent = null;
		double randomNumber = fitnessSum * Parameters.random.nextDouble();
		fitnessSum = 0;
		
		for (int i = 0; i < population.size(); i++) {
			fitnessSum += 1.0 / population.get(i).fitness;
			if (fitnessSum > randomNumber) {
				parent = population.get(i);
				break;
			}
		}
		
		return parent.copy();
		
	}

	/**
	 * Crossover / Reproduction
	 * 
	 * NEEDS REPLACED with proper method this code just returns exact copies of the
	 * parents. 
	 */
	private ArrayList<Individual> reproduce(Individual parent1, Individual parent2) {
		ArrayList<Individual> children = new ArrayList<>();
		//children.add(parent1.copy());
		//children.add(parent2.copy());
		
		Individual childOne = new Individual();
		Individual childTwo = new Individual();
		ArrayList<Double> childOneBiases = new ArrayList<Double>();
		ArrayList<Double> childTwoBiases = new ArrayList<Double>();
		
		ArrayList<Double> neuronOne = new ArrayList<Double>();
		ArrayList<Double> neuronTwo = new ArrayList<Double>();
		int neuronCounter = 0;
		int chromosomePointer = 0;
		
		for (int i = 0; i < Parameters.getNumHidden() * NeuralNetwork.numInput; i++) {
			chromosomePointer += 1;
			neuronOne.add(parent1.chromosome[i]);
			neuronTwo.add(parent2.chromosome[i]);
			if (i+1 % NeuralNetwork.numInput == 0) {
				neuronCounter += 1;
				int index = (Parameters.getNumHidden() - neuronCounter) * NeuralNetwork.numInput + neuronCounter;
				
				if (Parameters.random.nextBoolean()) {
					for (int j = 0; j < neuronOne.size(); j++) {
						childOne.chromosome[j] = neuronOne.get(j);
						childOneBiases.add(parent1.chromosome[i+index]);
						childTwo.chromosome[j] = neuronTwo.get(j);
						childTwoBiases.add(parent2.chromosome[i+index]);
					}
				}
				else {
					for (int j = 0; j < neuronOne.size(); j++) {
						childOne.chromosome[j] = neuronTwo.get(j);
						childOneBiases.add(parent2.chromosome[i+index]);
						childTwo.chromosome[j] = neuronOne.get(j);
						childTwoBiases.add(parent1.chromosome[i+index]);
					}
				}
				
				neuronOne.clear();
				neuronTwo.clear();
			}
		}
		
		for (int i = 0; i<childOneBiases.size(); i++) {
			childOne.chromosome[chromosomePointer] = childOneBiases.get(i);
			childTwo.chromosome[chromosomePointer] = childTwoBiases.get(i);
			chromosomePointer += 1;
		}
		
		childOneBiases.clear();
		childTwoBiases.clear();
		neuronCounter = 0;
		
		for (int i = chromosomePointer-1; i < parent1.chromosome.length; i++) {
			chromosomePointer += 1;
			neuronOne.add(parent1.chromosome[i]);
			neuronTwo.add(parent2.chromosome[i]);
			if (i+1 % Parameters.getNumHidden() == 0) {
				neuronCounter += 1;
				int index = (NeuralNetwork.numOutput - neuronCounter) * Parameters.getNumHidden() + neuronCounter;
				
				if (Parameters.random.nextBoolean()) {
					for (int j = 0; j < neuronOne.size(); j++) {
						childOne.chromosome[j] = neuronOne.get(j);
						childOneBiases.add(parent1.chromosome[i+index]);
						childTwo.chromosome[j] = neuronTwo.get(j);
						childTwoBiases.add(parent2.chromosome[i+index]);
					}
				}
				else {
					for (int j = 0; j < neuronOne.size(); j++) {
						childOne.chromosome[j] = neuronTwo.get(j);
						childOneBiases.add(parent2.chromosome[i+index]);
						childTwo.chromosome[j] = neuronOne.get(j);
						childTwoBiases.add(parent1.chromosome[i+index]);
					}
				}
				
				neuronOne.clear();
				neuronTwo.clear();
			}
		}
		
		for (int i = 0; i<childOneBiases.size(); i++) {
			childOne.chromosome[chromosomePointer] = childOneBiases.get(i);
			childTwo.chromosome[chromosomePointer] = childTwoBiases.get(i);
			chromosomePointer += 1;
		}
		
		children.add(childOne.copy());
		children.add(childTwo.copy());
		
		return children;
	} 
	
	/**
	 * Mutation
	 * 
	 * 
	 */
	private void mutate(ArrayList<Individual> individuals) {		
		for(Individual individual : individuals) {
			for (int i = 0; i < individual.chromosome.length; i++) {
				if (Parameters.random.nextDouble() < Parameters.mutateRate) {
					if (Parameters.random.nextBoolean()) {
						individual.chromosome[i] += (Parameters.mutateChange);
					} else {
						individual.chromosome[i] -= (Parameters.mutateChange);
					}
				}
			}
		}		
	}

	/**
	 * 
	 * Replaces the worst member of the population 
	 * (regardless of fitness)
	 * 
	 */
	private void replace(ArrayList<Individual> individuals) {
		for(Individual individual : individuals) {
			int idx = getWorstIndex();		
			population.set(idx, individual);
		}		
	}

	

	/**
	 * Returns the index of the worst member of the population
	 * @return
	 */
	private int getWorstIndex() {
		Individual worst = null;
		int idx = -1;
		for (int i = 0; i < population.size(); i++) {
			Individual individual = population.get(i);
			if (worst == null) {
				worst = individual;
				idx = i;
			} else if (individual.fitness > worst.fitness) {
				worst = individual;
				idx = i; 
			}
		}
		return idx;
	}	

	@Override
	public double activationFunction(double x) {
		if (x < -20.0) {
			return -1.0;
		} else if (x > 20.0) {
			return 1.0;
		}
		return Math.tanh(x);
	}
}
