package coursework;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
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
		
		double defaultMutateRate = 0.01D;
		double defaultMutateChange = 0.05D;
		double updatedMutateRate = 0.6D;
		double updatedMutateChange = 0.1D;
		
		Parameters.popSize = 50;
		Parameters.minGene = -3.0;
		Parameters.maxGene = 3.0;
		Parameters.mutateRate = defaultMutateRate;
		Parameters.mutateChange = defaultMutateChange;
		
		population = initialise();
		
		best = getBest();
		System.out.println("Best from initial population "+ best);
		
		boolean jump = false;
		double currentBest = Double.MAX_VALUE;
		int stuckCounter = 0;

		while (evaluations < Parameters.maxEvaluations) {
			int portion = 15;
			int randomLosers = 5;
			int randomMutations = 15;
			Collections.sort(population);

			ArrayList<Individual> bestOfCurretPopulation = new ArrayList<Individual>(population.subList(0, portion));
			for (int index = 0; index < randomLosers; index++) {
				int randomNumber = Parameters.random.nextInt(Parameters.popSize - portion) + portion;
				Individual chosenBadOne = population.get(randomNumber);

				while (exists(chosenBadOne, bestOfCurretPopulation)) {
					randomNumber = Parameters.random.nextInt(Parameters.popSize - portion) + portion;
					chosenBadOne = population.get(randomNumber);
				}
				bestOfCurretPopulation.add(chosenBadOne);
			}

			for (int index = 0; index < randomMutations; index++) {
				int randomNumber = Parameters.random.nextInt(bestOfCurretPopulation.size());
				Individual mutatedMember = mutate(bestOfCurretPopulation.get(randomNumber));

				while(exists(mutatedMember, bestOfCurretPopulation)) {
					mutatedMember = mutate(mutatedMember);
				}
				mutatedMember.fitness = Fitness.evaluate(mutatedMember, this);
				bestOfCurretPopulation.add(mutatedMember);
			}

			breed(bestOfCurretPopulation);
			
			best = getBest();

			outputStats();
			
			if (best.fitness < 0.015D && evaluations >= (Parameters.maxEvaluations / 2)) {
				evaluations = Parameters.maxEvaluations + 1;
			}
			
			meanScore();
			
			if (best.fitness < currentBest) {
				currentBest = best.fitness;
				stuckCounter = 0;
			} else {
				stuckCounter += 1;
			}
			
			if((((meanScore() - best.fitness) < 0.01) || stuckCounter >= 50)) {
				System.out.println(stuckCounter);
				Parameters.mutateChange = updatedMutateChange;
				Parameters.mutateRate = updatedMutateRate;
				jump = true;
			}
			else if(jump) {
				Parameters.mutateChange = defaultMutateChange;
				Parameters.mutateRate = defaultMutateRate;
				jump = false;
			}
		}
		System.out.println(population);
		
		saveNeuralNetwork();
	}
	
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
	
	private ArrayList<Individual> initialise() {
		population = new ArrayList<>();
		for (int i = 0; i < Parameters.popSize; ++i) {
			//chromosome weights are initialised randomly in the constructor
			Individual individual = new Individual();
			while(exists(individual, population)) {
				individual = new Individual();
			}
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
	
	double reduceFactor = 1.0D;
	private Individual mutate(Individual individual) {
		Individual newMember = new Individual();
		newMember.chromosome = Arrays.copyOf(individual.chromosome, individual.chromosome.length);
		
		for (int i = 0; i < newMember.chromosome.length; i++) {
			if (Parameters.random.nextDouble() < Parameters.mutateRate) {
				if(Parameters.random.nextBoolean()) {
					newMember.chromosome[i] += (Parameters.mutateChange*reduceFactor);
				}
				else {
					newMember.chromosome[i] -= (Parameters.mutateChange*reduceFactor);
				}
			}
		}
//		reduceFactor*=0.99999D;
//		newMember.fitness = Fitness.evaluate(newMember, this);
		return newMember.copy();
	}
	
	private Boolean exists(Individual ind, ArrayList<Individual> currentPopulation) {		
		for (Individual temp : currentPopulation) {
			if (Arrays.equals(temp.chromosome, ind.chromosome)) {
				return true;
			}
		}
		return false;
	}
	
	private void breed(ArrayList<Individual> currentBestPopulation){
		population.clear();
		population.addAll(currentBestPopulation);

		while (population.size() < Parameters.popSize) {
			Individual parent1 = select(currentBestPopulation); 
			Individual parent2 = select(currentBestPopulation);

			while (Arrays.equals(parent1.chromosome, parent2.chromosome)) {
				parent2 = select(currentBestPopulation);
			}
		
			ArrayList<Individual> children = reproduce(parent1, parent2);
			ArrayList<Individual> updatedChildren = new ArrayList<>();
			
			for (Individual child : children) {
				while (exists(child, population)) {
					child = mutate(child);
				}
				updatedChildren.add(child);
			}
			
			evaluateIndividuals(updatedChildren);
			
			population.addAll(updatedChildren);
		}
	}
	
	private Individual tournamentSelection(ArrayList<Individual> currentBestPopulation) {
		// The tournament size of 10% of the population
		int tournamentSize = 2;//(int)(currentBestPopulation.size()*0.1);
		TreeMap<Integer, Individual> potentialParents = new TreeMap<Integer, Individual>();

		while(potentialParents.size() < tournamentSize) {
			int randomIndex = Parameters.random.nextInt(currentBestPopulation.size());
			potentialParents.put(randomIndex, currentBestPopulation.get(randomIndex));
		}

		double bestFitness = Double.MAX_VALUE;
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

		return chosenParent.copy();
	}
	
	private Individual rouletteSelection(ArrayList<Individual> currentPopulation) {
		double fitnessSum = 0;
		for (int i = 0; i < currentPopulation.size(); i++) {
			fitnessSum += (1.0 - currentPopulation.get(i).fitness);
		}

		Individual parent = null;

		double randomNumber = fitnessSum * Parameters.random.nextDouble();
		fitnessSum = 0;

		for (int i = 0; i < currentPopulation.size(); i++) {
			fitnessSum += (1.0 - currentPopulation.get(i).fitness);
			if (fitnessSum > randomNumber) {
				parent = currentPopulation.get(i);
				break;
			}
		}
		return parent;
	}

	private Individual select(ArrayList<Individual> currentBestPopulation) {
		Individual parent = null;

		/* Selects based on tournament */
		if(Parameters.random.nextDouble() < 0.5) {
			parent = tournamentSelection(currentBestPopulation);
		} else {
			parent = rouletteSelection(currentBestPopulation);
		}

		return parent.copy();
	}
	
	private ArrayList<Individual> reproduce(Individual parent1, Individual parent2) {

		ArrayList<Individual> children = new ArrayList<>();

		/* One point crossover */
		// This assumes that parent1 and parent2 have chromosomes with the same length
//		int cutPoint = NeuralNetwork.numInput*Parameters.getNumHidden()+Parameters.getNumHidden();
		int cutPoint = Parameters.random.nextInt(parent1.chromosome.length);
//		int cutPoint = parent1.chromosome.length/2;
		onePointCrossOver(children, parent1, parent2, cutPoint);
		
		/* Uniform crossover */
//		uniformCrossover(children, parent1, parent2);
		
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
		
		childOne = mutate(childOne);
//		childTwo = mutate(childTwo);

		children.add(childOne);
//		children.add(childTwo);
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

		childOne = mutate(childOne);
		
		children.add(childOne);
//		children.add(childTwo);
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
	
	private double meanScore() {
		double totalFitness = 0;
		for (Individual ind : population) {
			totalFitness += ind.fitness;
		}
		double meanScore = totalFitness/population.size();
		System.out.println("\tMean: " + meanScore);
		return meanScore;
	}
}
