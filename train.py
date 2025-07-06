from linear_regression import LinearRegression
import argparse
import os.path
import signal
import sys
import json

def termination_handler(signum, frame):
    print("Termination requested...")
    sys.exit()

def arguments_configuration():
    parser = argparse.ArgumentParser(
        prog='ft_linear_regression',
        description='This programm trains a linear regression using the Gradient Descent algorithm')

    parser.add_argument('filename')

    data_input = parser.add_argument_group('Data input options', 'Parameters when reading the data')
    data_input.add_argument('--delimiter', '-d', default=',', help='Delimiter for data input')
    data_input.add_argument('--skip_header', default=1, help='Skip header for data input')

    algoritm = parser.add_argument_group('Algorithm options', 'Constants and initial values for the Gradient Descent algorithm')
    algoritm.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='Learning rate')
    algoritm.add_argument('--minimum_step_size', '-mss', type=float, default=0.0001, help='Minimun step size')
    algoritm.add_argument('--steps', '-s', type=int, default=1000, help='Maximun number of steps')
    algoritm.add_argument('--init_theta0', '-th0', type=float, default=0, help='Initial value of theta0')
    algoritm.add_argument('--init_theta1', '-th1', type=float, default=0, help='Initial value of theta1')

    data_input.add_argument('--plot', '-p', action='store_true', help='Show graphical results')
    args = parser.parse_args()
    print(args)
    return args

signal.signal(signal.SIGINT, termination_handler)
signal.signal(signal.SIGTERM, termination_handler)

args = arguments_configuration()
file_path = os.path.abspath(args.filename)

l = LinearRegression(file_path, args)
l.print_results()
if args.plot:
    l.plot_result()