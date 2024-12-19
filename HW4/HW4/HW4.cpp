#include "HW4.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <set>
#include <thread>
#include <chrono>

double driftSDE(double x, double const b)
{
	return -b;
}

class ItoProcess
{
private:
	double (*drift)(double, double);

	// Simulation parameters
	int max_iter;          // maximum number of iterations
	double dt;      // Time step
	int seed;              // Seed for random number generator
	double b;

	// Random number generator
	std::mt19937 generator;
	std::normal_distribution<double> normal_dist;
	std::gamma_distribution<double> gamma_dist;


public:
	ItoProcess(double (*drift)(double, double), int max_iter, double timestep, double b, unsigned int seed = 0) :
		drift(drift),
		max_iter(max_iter),
		dt(timestep),
		b(b),
		generator(seed),
		normal_dist(0.0, 1.0),
		gamma_dist(2.0, 1 / b)
	{
		generator.seed(std::random_device{}());
	}

	std::vector<double> simulate_path()
	{
		std::vector<double> path(max_iter);
		double x0 = gamma_dist(generator);
		path[0] = x0;

		for (int i = 1; i < max_iter; i++)
		{
			double dW = normal_dist(generator) * sqrt(dt);
			path[i] = path[i - 1] + drift(path[i - 1], b) * dt + dW;
			if (path[i] <= 0)
			{
				break;
			}
		}
		return path;
	}

	double simulate_time(std::mt19937& gen,
		std::normal_distribution<double>& normal_dist,
		std::gamma_distribution<double>& gamma_dist)
	{
		double x0 = gamma_dist(gen);

		double x1;
		for (int i = 1; i < max_iter; i++)
		{
			double dW = normal_dist(gen) * sqrt(dt);
			x1 = x0 - b * dt + dW;

			if (x1 <= 0) {
				return i * dt;  // Return actual hitting time
			}
			x0 = x1;
		}

		// If never hits zero, what should we return?
		// Returning x1 is not correct since it's a position, not a time.
		// Maybe return max_iter * dt or some sentinel value.
		return max_iter * dt;
	}


	std::vector<double> run_simulation(int m)
	{
		std::mt19937 local_gen(std::random_device{}());
		std::normal_distribution<double> local_normal_dist(0.0, 1.0);
		std::gamma_distribution<double> local_gamma_dist(2.0, 1.0 / b);
		std::vector<double> times(m);
		for (int i = 0; i < m; i++)
		{
			times[i] = simulate_time(local_gen, local_normal_dist, local_gamma_dist);
		}

		std::sort(times.begin(), times.end());
		return times;
	}

	std::vector<std::vector<double>> get_probabilities(int m)
	{
		std::vector<double> times = run_simulation(m);
		int max_time_steps = static_cast<int>(times.back() / dt);

		std::set<double> unique_times(times.begin(), times.end());
		int number_of_unique_times = unique_times.size();

		std::vector<double> probabilities(max_time_steps);
		probabilities[0] = 1.0;
		int times_counter = 0;

		for (int i = 1; i < max_time_steps; i++) {
			if (times[times_counter] <= i * dt) {
				double n = 0;

				while (times[times_counter] <= i * dt) {
					times_counter++;
					n++;
				}
				probabilities[i] = probabilities[i - 1] - n / m;
			}
			else {
				probabilities[i] = probabilities[i - 1];
			}
		}

		// Create the matrix with unique times and their corresponding probabilities
		std::vector<std::vector<double>> result;
		for (const auto& time : unique_times) {
			int index = static_cast<int>(time / dt);
			if (index < max_time_steps) {
				result.push_back({ time, probabilities[index] });
			}
		}

		return result;
	}


	std::vector<std::vector<double>> get_probabilities_in_parallel(int m, int num_threads) {

		unsigned int num_threads_max = std::thread::hardware_concurrency();
		if (num_threads == 0) {
			num_threads = num_threads_max;
		}
		else if (num_threads > num_threads_max) {
			num_threads = num_threads_max;
		}

		int chunk_size = m / num_threads;
		std::vector<std::vector<double>> all_times(num_threads);

		auto worker = [&](int start, int end, std::vector<double>& times_out, int thread_id) {
			// Create a private RNG for each thread
			std::mt19937 local_gen(std::random_device{}());
			std::normal_distribution<double> local_normal_dist(0.0, 1.0);
			std::gamma_distribution<double> local_gamma_dist(2.0, 1.0 / b);

			for (int i = start; i < end; ++i) {
				times_out.push_back(simulate_time(local_gen, local_normal_dist, local_gamma_dist));
			}
			std::cout << "Thread " << thread_id << " finished." << std::endl;
			};


		// Launch threads
		std::vector<std::thread> threads;
		for (unsigned int t = 0; t < num_threads; ++t) {
			int start = t * chunk_size;
			int end = (t == num_threads - 1) ? m : start + chunk_size; // Last thread takes remaining
			threads.emplace_back(worker, start, end, std::ref(all_times[t]), t);
		}

		// Wait for all threads to finish
		for (auto& t : threads) {
			t.join();
		}

		// Combine results
		std::vector<double> combined_times;
		for (const auto& times : all_times) {
			combined_times.insert(combined_times.end(), times.begin(), times.end());
		}

		// Sort combined times
		std::sort(combined_times.begin(), combined_times.end());

		// Calculate probabilities
		int max_time_steps = static_cast<int>(combined_times.back() / dt);
		std::vector<double> probabilities(max_time_steps);
		probabilities[0] = 1.0;

		int times_counter = 0;
		for (int i = 1; i < max_time_steps; ++i) {
			if (combined_times[times_counter] <= i * dt) {
				double n = 0;
				while (combined_times[times_counter] <= i * dt) {
					times_counter++;
					n++;
				}
				probabilities[i] = probabilities[i - 1] - n / m;
			}
			else {
				probabilities[i] = probabilities[i - 1];
			}
		}

		// Create the matrix with unique times and their corresponding probabilities
		std::set<double> unique_times(combined_times.begin(), combined_times.end());
		std::vector<std::vector<double>> result;
		for (const auto& time : unique_times) {
			int index = static_cast<int>(time / dt);
			if (index < max_time_steps) {
				result.push_back({ time, probabilities[index] });
			}
		}

		return result;
	}



	void save_vec_to_csv(std::vector<double> vec, std::string filename)
	{
		std::ofstream file;
		file.open(filename);
		for (int i = 0; i < vec.size(); i++)
		{
			file << vec[i] << std::endl;
		}
		file.close();
	}
};


int main()
{
	unsigned int num_threads = std::thread::hardware_concurrency();
	std::cout << "Number of threads available: " << num_threads << std::endl;

	ItoProcess process(driftSDE,
		1000,	// max_iter
		0.01,	// dt	
		1,		// b
		0);		// seed

	auto start = std::chrono::high_resolution_clock::now();

	std::vector<std::vector<double>> result = process.get_probabilities_in_parallel(1e6, 10);

	// End timing
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = end - start;

	std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

	// Save the matrix to a CSV file
	std::ofstream file("probabilities.csv");
	for (const auto& row : result) {
		file << row[0] << "," << row[1] << std::endl;
	}
	file.close();

	return 0;
};