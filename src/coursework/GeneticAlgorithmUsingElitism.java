package coursework;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.TreeMap;

import com.sun.org.apache.xalan.internal.xsltc.runtime.Parameter;

import java.util.Map.Entry;

import model.Fitness;
import model.Individual;
import model.NeuralNetwork;

public class GeneticAlgorithmUsingElitism extends NeuralNetwork {
	
	@Override
	public void run() {
		System.out.println("Running GeneticAlgorithmUsingElitism");
		
		Parameters.popSize = 50;
		Parameters.minGene = -3.0;
		Parameters.maxGene = 3.0;
		Parameters.mutateRate = 0.01;
		Parameters.mutateChange = 0.05;
		
		population = initialise();
		
		best = getBest();
		System.out.println("Best from initial population "+ best);
		
		while (evaluations < Parameters.maxEvaluations) {
			int portion = (int)(population.size() * 0.1); // 5
			int randomLosers = 2;
			int randomMutations = 3;
			Collections.sort(population);
			
			ArrayList<Individual> bestOfCurretPopulation = new ArrayList<Individual>(population.subList(0, portion));
			
			for (int index = 0; index < randomLosers; index++) {
				int randomNumber = Parameters.random.nextInt(Parameters.popSize - portion) + portion;
				bestOfCurretPopulation.add(population.get(randomNumber));
			}

			for (int index = 0; index < randomMutations; index++) {
				int randomNumber = Parameters.random.nextInt((bestOfCurretPopulation.size()));
				bestOfCurretPopulation.add(mutate(bestOfCurretPopulation.get(randomNumber)));
				System.out.println(randomNumber + bestOfCurretPopulation.get(randomNumber).toString());
			}
			
			System.out.println(population.toString());
			System.out.println(bestOfCurretPopulation.toString());
			System.exit(0);
			
			population = breed(bestOfCurretPopulation);		
//			System.out.println(population);
			best = getBest();
			
			outputStats();
			
			if (best.fitness < 0.015D) {
				evaluations = Parameters.maxEvaluations + 1;
			}
		}
		
		saveNeuralNetwork();
	}
	
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
	
	private void evaluateIndividuals(ArrayList<Individual> individuals) {
		for (Individual individual : individuals) {
			individual.fitness = Fitness.evaluate(individual, this);
		}
	}
	
	private Individual getBest() {
		best = null;
		for (Individual individual : population) {
			if (best == null) {
				best = individual.copy();
			} else if (individual.fitness < best.fitness) {
				best = individual.copy();
			}
		}
		return best;
	}
	
	float reduceFactor = 1.0f;
	private Individual mutate(Individual individual) {
		Individual newMember = new Individual();
		newMember.chromosome = individual.chromosome;
		
		for (int i = 0; i < newMember.chromosome.length; i++) {
			if (Parameters.random.nextDouble() < Parameters.mutateRate) {
				if(Parameters.random.nextBoolean()) {
					newMember.chromosome[i] += (Parameters.mutateChange);
				}
				else {
					newMember.chromosome[i] += (Parameters.mutateChange);
				}
			}
		}
//		reduceFactor*=0.9998f;
		newMember.fitness = Fitness.evaluate(newMember, this);
		return newMember.copy();
	}
	
	private ArrayList<Individual> breed(ArrayList<Individual> currentBestPopulation){
		ArrayList<Individual> newPopulation = new ArrayList<Individual>();
		
		for (Individual x : currentBestPopulation) {
			newPopulation.add(x);
		}
		
		while (newPopulation.size() < Parameters.popSize) {
			Individual parent1 = select(currentBestPopulation); 
			Individual parent2 = select(currentBestPopulation);
		
			if (parent1 == parent2) {
				System.out.print("\n\tOuch, the same parent\n");
			}
		
			ArrayList<Individual> children = reproduce(parent1, parent2);
		
			evaluateIndividuals(children);
		
			for (Individual x : children) {
				newPopulation.add(x);
			}
		}
		System.out.println(newPopulation.size());
		return newPopulation;
	}
	
	private Individual tournamentSelection(ArrayList<Individual> currentBestPopulation) {
		// The tournament size of 10% of the population
		int tournamentSize = 2;//(int)(currentBestPopulation.size()*0.1);
		TreeMap<Integer, Individual> potentialParents = new TreeMap<Integer, Individual>();

		while(potentialParents.size() < tournamentSize) {
			int randomIndex = Parameters.random.nextInt(currentBestPopulation.size());
			potentialParents.put(randomIndex, currentBestPopulation.get(randomIndex));
		}

		double bestFitness = Integer.MAX_VALUE;
		Individual chosenParent = null;

		for(Entry<Integer, Individual> entry : potentialParents.entrySet()) {
			Individual individual = entry.getValue();
			double currentFitness = individual.fitness;

			if (chosenParent == null) {
				chosenParent = individual;
				bestFitness = currentFitness;
			} else if(bestFitness > currentFitness) {
				chosenParent = individual;
				bestFitness = currentFitness;
			}
		}

		return chosenParent;
	}

	private Individual select(ArrayList<Individual> currentBestPopulation) {
		Individual parent = null;

		/* Selects based on tournament */
		parent = tournamentSelection(currentBestPopulation);

		return parent.copy();
	}
	
	private ArrayList<Individual> reproduce(Individual parent1, Individual parent2) {

		ArrayList<Individual> children = new ArrayList<>();

		/* One point crossover */
		// This assumes that parent1 and parent2 have chromosomes with the same length
		//int cutPoint = NeuralNetwork.numInput*Parameters.getNumHidden()+Parameters.getNumHidden();
		//onePointCrossOver(children, parent1, parent2, cutPoint);
		
		/* Uniform crossover */
		uniformCrossover(children, parent1, parent2);
		
		/*DEBUG INF0*/
//		System.out.println("parent1: "+Arrays.toString(parent1.chromosome));
//		System.out.println("parent2: "+Arrays.toString(parent2.chromosome));
//		System.out.println("childOne After: "+Arrays.toString(children.get(0).chromosome));
//		System.out.println("childTwo After: "+Arrays.toString(children.get(1).chromosome));
//		System.exit(0);
		/***********************************/

		return children;
	} 
	
	private void onePointCrossOver(ArrayList<Individual> children, Individual parent1, Individual parent2, int cutPoint) 
	{
		Individual childOne = new Individual();
		Individual childTwo = new Individual();

		// This assumes that parent1 and parent2 have chromosomes with the same length
		int chromosomeLength = parent1.chromosome.length;

		for(int i = 0; i < cutPoint; i++) {
			childOne.chromosome[i] = parent1.chromosome[i];
			childTwo.chromosome[i] = parent2.chromosome[i];
		}

		for(int i = cutPoint; i < chromosomeLength; i++) {
			childOne.chromosome[i] = parent2.chromosome[i];
			childTwo.chromosome[i] = parent1.chromosome[i];
		}

		children.add(childOne);
		children.add(childTwo);
	}
	
	private void uniformCrossover(ArrayList<Individual> children, Individual parent1, Individual parent2) 
	{
		Individual childOne = new Individual();
		Individual childTwo = new Individual();

		// This assumes that parent1 and parent2 have chromosomes with the same length
		int chromosomeLength = parent1.chromosome.length;

		for(int i = 0; i < chromosomeLength; i++) {
			if (Parameters.random.nextDouble() < 0.5D) {
				childOne.chromosome[i] = parent1.chromosome[i];
			}
			else {
				childOne.chromosome[i] = parent2.chromosome[i];
			}
		}

		for(int i = 0; i < chromosomeLength; i++) {
			if (Parameters.random.nextDouble() < 0.5D) {
				childTwo.chromosome[i] = parent1.chromosome[i];
			}
			else {
				childTwo.chromosome[i] = parent2.chromosome[i];
			}
		}

		
		children.add(childOne);
		children.add(childTwo);
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
