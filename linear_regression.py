import numpy as np
import matplotlib.pyplot as ptl
import sys
import json

class LinearRegression():

    def __init__(self, file_path, args):
        self.file_path = file_path
        print(self.file_path)
        self.args = args
        self.get_data()
        self.get_standarized_data()
        self.gradient_descent()
        self.destandarize()
        self.save_to_json()
        print(f"theta0 sobre valores estandarizados: {self.theta0}")
        print(f"theta1 sobre valores estandarizados: {self.theta1}")

    def read_file(self):
        try:
            self.data = np.genfromtxt(self.file_path, delimiter=self.args.delimiter, skip_header=self.args.skip_header)
        except FileNotFoundError:
            print(f'Error: couldn´t find file {self.file_path}')
            sys.exit(1)
        except PermissionError:
            print(f"Permission denied to access {self.file_path}")
            sys.exit(1)
        except Exception as e:
            print(f"An exception type {type(e).__name__} has ocurred, please check the input file1")
            sys.exit(1)

    def get_data(self):
        try:
            self.read_file()
            if self.data.size == 0:
                print(f"The file {self.file_path} is empty")
                sys.exit(1)
            elif self.data.shape[0] < 2:
                print(f"The file {self.file_path} must have at least 2 rows")
                sys.exit(1)
            elif self.data.shape[1] != 2:
                print(f"The file {self.file_path} must have 2 columns")
                sys.exit(1)
            if np.isnan(self.data).any():
                self.data = self.data[~np.isnan(self.data).any(axis=1)]
        except Exception as e:
            print(f"An exception type {type(e).__name__} has ocurred, please check the input file2")
            sys.exit(1)

    def get_standarized_data(self):
        #ddof=1 calcula la std muestral en vez de la std poblacional, lo que segun chatgpt 
        # es mas adecuado para este proyecto porque tenemos una muestra que no 
        # representa todos los coches del mundo
        self.std_x = np.std(self.data, axis = 0, ddof=1)[0]
        self.mean_x = np.mean(self.data, axis=0)[0]
        self.standarized_data = self.data.copy()
        self.standarized_data[:,0] = (self.standarized_data[:,0] - self.mean_x) / self.std_x

    def update_thetas(self, total_variability):
        m = len(self.standarized_data[:,0])
        error_array = self.standarized_data[:,0] * self.theta1 + self.theta0 - self.standarized_data[:,1]
        theta1_array = error_array * self.standarized_data[:,0]
        self.medium_square_error.append(np.mean(error_array**2))
        residual_cuadratic_error = np.sum(error_array**2)
        self.coefficient_of_determination.append(1 - residual_cuadratic_error / total_variability)
        total_error = np.sum(error_array)
        new_theta0 = self.theta0 - self.args.learning_rate * total_error / m
        new_theta1 = self.theta1 - self.args.learning_rate * np.sum(theta1_array) / m
        return new_theta0, new_theta1

    def gradient_descent(self):
        self.theta0 = self.args.init_theta0
        self.theta1 = self.args.init_theta1
        self.medium_square_error = []
        self.coefficient_of_determination = []
        total_variability = np.sum((self.standarized_data[:,1] - np.mean(self.standarized_data[:,1]))**2)
        for step in range(self.args.steps):
            new_theta0, new_theta1 = self.update_thetas(total_variability)
            if abs(new_theta0 - self.theta0) < self.args.minimum_step_size and abs(new_theta1 - self.theta1) < self.args.minimum_step_size:
                break
            self.theta0, self.theta1 = new_theta0, new_theta1
        error_array = self.standarized_data[:,0]*self.theta1 + self.theta0 - self.standarized_data[:,1]
        self.medium_square_error.append((np.mean(error_array))**2)

    def destandarize(self):
        self.theta1 = self.theta1 / self.std_x
        self.theta0 = self.theta0 - self.theta1 * self.mean_x
        self.mse = np.mean((self.data[:,1] - self.theta0 - self.theta1 * self.data[:,0]) ** 2)
        total_variability = np.sum((self.data[:,1] - np.mean(self.data[:,1])) ** 2)
        residual_variability = np.sum((self.data[:,1] - self.theta0 - self.theta1 * self.data[:,0]) ** 2)
        self.r2 = 1 - residual_variability / total_variability

    def save_to_json(self):
        results = {'theta0':self.theta0, 'theta1':self.theta1, 'medium_square_error': self.mse, 'coefficient_of_determination': self.r2}
        with open('results.json', 'w') as jsonfile:
            json.dump(results, jsonfile)

    def print_results(self):
        print("Parameters resulting of training the model:")
        print(f"theta0: {self.theta0}")
        print(f"theta1: {self.theta1}")
        print(f"medium_square_error: {self.mse}")
        print(f"coefficient_of_determination: {self.r2}")

    def plot_result(self):
        data_sorted = self.data[self.data[:, 0].argsort()]
        y_max = (np.max(data_sorted[:,1])) * 1.1
        x_max = (np.max(data_sorted[:,0])) * 1.1
        y_predicted = data_sorted[:,0] * self.theta1 + self.theta0
        ptl.rcParams["figure.figsize"] = (25, 10)

        ptl.subplot(1, 3, 1)
        ptl.plot(data_sorted[:,0], data_sorted[:,1], ls='--', marker='o')
        ptl.plot(data_sorted[:,0], y_predicted)
        ptl.ylim(0, y_max)
        ptl.xlim(0, x_max)
        ptl.title('Price vs mileage')
        ptl.xlabel('mileage')
        ptl.ylabel('price')
        ptl.legend(['input', 'linear regression'], loc=0)

        ptl.subplot(1, 3, 2)
        ptl.plot(self.medium_square_error, marker='o', markersize=2)
        # ptl.plot(coeffient_of_determination)
        ptl.title('Medium square error')
        ptl.xlabel('steps')
        ptl.ylabel('mse')

        ptl.subplot(1, 3, 3)
        ptl.plot(self.coefficient_of_determination, marker='o', markersize=2)
        # ptl.plot(coeffient_of_determination)
        ptl.title('coefficient_of_determination')
        ptl.xlabel('steps')
        ptl.ylabel('R_square')
        ptl.show()