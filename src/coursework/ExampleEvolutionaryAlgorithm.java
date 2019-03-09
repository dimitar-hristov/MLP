package coursework;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

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
	/*
	 * TODO - Replace more chomosomes from the population, e.g. 10% using for loop
	 * 		- Check reproduce if it works correctly
	 * 		- Change activation function to be Adam
	 * 		- Only replace if the new individual has better fitness score than the worst in the current population
	 * 		- Make sure the parents are different individuals
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
	
	private int reproduceHelper(int offset, int startIndex, Integer chromosemeCounter, Individual mother, Individual father, Individual childOne, Individual childTwo, int range, int previousLayerSize, int curretLayerSize) {
		int neuronCounter = 0;
		ArrayList<Double> neuronFather = new ArrayList<Double>();
		ArrayList<Double> neuronMother = new ArrayList<Double>();
		ArrayList<Double> childOneBiases = new ArrayList<Double>();
		ArrayList<Double> childTwoBiases = new ArrayList<Double>();
		Random bool = new Random();
		for (int i = startIndex; i < range; i++) {
//			System.out.print(i+";");
			neuronFather.add(father.chromosome[i]);
			neuronMother.add(mother.chromosome[i]);
			if ((i+1) % previousLayerSize == 0) {
				neuronCounter += 1;
				int index = (curretLayerSize - neuronCounter) * previousLayerSize + neuronCounter;
				
				if (bool.nextBoolean()) {
					for (int j = 0; j < neuronFather.size(); j++) {
						childOne.chromosome[j+((neuronCounter-1)*previousLayerSize)+offset] = neuronFather.get(j);
						childTwo.chromosome[j+((neuronCounter-1)*previousLayerSize)+offset] = neuronMother.get(j);
						chromosemeCounter += 1;
					}
					childOneBiases.add(father.chromosome[i+index]);
					childTwoBiases.add(mother.chromosome[i+index]);
				}
				else {
					for (int j = 0; j < neuronFather.size(); j++) {
						childOne.chromosome[j+((neuronCounter-1)*previousLayerSize)+offset] = neuronMother.get(j);
						childTwo.chromosome[j+((neuronCounter-1)*previousLayerSize)+offset] = neuronFather.get(j);
						chromosemeCounter += 1;
					}
					childOneBiases.add(mother.chromosome[i+index]);
					childTwoBiases.add(father.chromosome[i+index]);
				}
				
				neuronFather.clear();
				neuronMother.clear();
			}
		}
		
		for (int i = 0; i<childOneBiases.size(); i++) {
//			System.out.print(childOneBiases.size()+";");
			childOne.chromosome[chromosemeCounter] = childOneBiases.get(i);
			childTwo.chromosome[chromosemeCounter] = childTwoBiases.get(i);
			chromosemeCounter += 1;
		}
		return chromosemeCounter;
	}

	/**
	 * Crossover / Reproduction
	 * 
	 * NEEDS REPLACED with proper method this code just returns exact copies of the
	 * parents. 
	 */
	private ArrayList<Individual> reproduce(Individual parent1, Individual parent2) {
		ArrayList<Individual> children = new ArrayList<>();
		
		Individual childOne = new Individual();
		Individual childTwo = new Individual();
		
		Integer chromosomeCounter = new Integer(0);
		
		int range = Parameters.getNumHidden() * NeuralNetwork.numInput;
//		System.out.println("parent1: "+Arrays.toString(parent1.chromosome));
//		System.out.println("parent2: "+Arrays.toString(parent2.chromosome));
//		System.out.println("childOne Before: "+Arrays.toString(childOne.chromosome));
//		System.out.println("childTwo Before: "+Arrays.toString(childTwo.chromosome));
		int result = reproduceHelper(0, 0, chromosomeCounter, parent1, parent2, childOne, childTwo, range, NeuralNetwork.numInput, Parameters.getNumHidden());
		range = parent1.chromosome.length;
		reproduceHelper(result, result, result, parent1, parent2, childOne, childTwo, range, Parameters.getNumHidden(), NeuralNetwork.numOutput);
//		System.out.println("childOne After: "+Arrays.toString(childOne.chromosome));
//		System.out.println("childTwo After: "+Arrays.toString(childTwo.chromosome));
		
		children.add(childOne.copy());
		children.add(childTwo.copy());
//		System.exit(0);
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
