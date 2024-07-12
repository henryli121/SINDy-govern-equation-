import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from matplotlib import rcParams

# Set up LaTeX formatting
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts} \usepackage{multicol}'
rcParams['legend.handleheight'] = 3.0

# Load the data
Y_final = np.load("Y_final.npy")
Y1 = Y_final[:, 0]  # First feature column
Y2 = Y_final[:, 1]  # Second feature column
Y3 = Y_final[:, 2]  # Third feature column
time_steps = np.arange(Y_final.shape[0])

# Split the data into three intervals
t1 = time_steps[0:10]
t2 = time_steps[10:600]
t3 = time_steps[600:2000]

Y1_intervals = [Y1[0:10], Y1[10:600], Y1[600:2000]]
Y2_intervals = [Y2[0:10], Y2[10:600], Y2[600:2000]]
Y3_intervals = [Y3[0:10], Y3[10:600], Y3[600:2000]]

# Build the simplified library of candidate functions
def build_solution_library(t):
    library = []
    library.append(np.ones(len(t)))      # Constant term
    library.append(t)                    # Linear term
    library.append(t**2)                 # Quadratic term
    library.append(np.exp(-t))           # Exponential term
    library.append(np.sin(t))            # Sine term
    library.append(np.cos(t))            # Cosine term
    library.append(t * np.exp(-t))       # Interaction term
    return np.vstack(library).T

Theta1 = build_solution_library(t1)
Theta2 = build_solution_library(t2)
Theta3 = build_solution_library(t3)

def fit_lasso(Theta, Y):
    # Scale the features
    scaler = StandardScaler()
    Theta_scaled = scaler.fit_transform(Theta)

    # Solve the sparse regression problem using cross-validation to find the optimal alpha
    lasso_cv = LassoCV(cv=5, fit_intercept=True, max_iter=10000000, alphas=np.logspace(-6, 1, 100))
    lasso_cv.fit(Theta_scaled, Y)
    coefficients = lasso_cv.coef_
    intercept = lasso_cv.intercept_

    return coefficients, intercept, scaler

def construct_solution(t, coefficients, intercept, scaler):
    Theta = build_solution_library(t)
    Theta_scaled = scaler.transform(Theta)
    return intercept + np.dot(Theta_scaled, coefficients)

terms = ['1', 't', 't^2', 'exp(-t)', 'sin(t)', 'cos(t)', 't*exp(-t)']

# Fit and construct solutions for all intervals and features
Y_intervals = [Y1_intervals, Y2_intervals, Y3_intervals]
identified_solutions = [[], [], []]

for i, Y_interval in enumerate(Y_intervals):
    for j, (Y, t) in enumerate(zip(Y_interval, [t1, t2, t3])):
        coefficients, intercept, scaler = fit_lasso(eval(f'Theta{j+1}'), Y)
        identified_y = construct_solution(t, coefficients, intercept, scaler)
        identified_solutions[i].append(identified_y)

# Correctly slice time_steps for plotting
time_steps1 = time_steps[0:10]
time_steps2 = time_steps[10:600]
time_steps3 = time_steps[600:2000]

# Combine the piecewise solutions
identified_y1 = np.concatenate(identified_solutions[0])
identified_y2 = np.concatenate(identified_solutions[1])
identified_y3 = np.concatenate(identified_solutions[2])

# Plot the results
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plot for Y1
axs[0, 0].plot(time_steps, Y1, label='Actual $y_1(t)$')
axs[0, 0].plot(np.concatenate([time_steps1, time_steps2, time_steps3]), identified_y1, '--', label='Identified $y_1(t)$')
axs[0, 0].legend()
axs[0, 0].set_xlabel('Time')
axs[0, 0].set_ylabel('$y_1(t)$')
axs[0, 0].set_title('Actual vs Identified $y_1(t)$')

# Plot for Y2
axs[0, 1].plot(time_steps, Y2, label='Actual $y_2(t)$')
axs[0, 1].plot(np.concatenate([time_steps1, time_steps2, time_steps3]), identified_y2, '--', label='Identified $y_2(t)$')
axs[0, 1].legend()
axs[0, 1].set_xlabel('Time')
axs[0, 1].set_ylabel('$y_2(t)$')
axs[0, 1].set_title('Actual vs Identified $y_2(t)$')

# Plot for Y3
axs[1, 0].plot(time_steps, Y3, label='Actual $y_3(t)$')
axs[1, 0].plot(np.concatenate([time_steps1, time_steps2, time_steps3]), identified_y3, '--', label='Identified $y_3(t)$')
axs[1, 0].legend()
axs[1, 0].set_xlabel('Time')
axs[1, 0].set_ylabel('$y_3(t)$')
axs[1, 0].set_title('Actual vs Identified $y_3(t)$')

# Manually input the piecewise functions
piecewise_functions = [
    r'$y_1(t) = \left\{ \begin{array}{ll} '
    r'12.510352 + 0.151419 \cdot t^2 - 8.621061 \cdot e^{-t} + 1.289994 \cdot \sin(t) - 3.172733 \cdot \cos(t) + 70.396992 \cdot t \cdot e^{-t}, & \quad t \in [0, 10) \\ '
    r'40.759198 + 0.067420 \cdot t - 0.000126 \cdot t^2 - 125056.595936 \cdot e^{-t} + 0.000045 \cdot \sin(t) + 0.002825 \cdot \cos(t), & \quad t \in [10, 600) \\ '
    r'34.852633 + 0.007129 \cdot t - 0.000002 \cdot t^2, & \quad t \in [600, 2000] '
    r'\end{array} \right.$',

    r'$y_2(t) = \left\{ \begin{array}{ll} '
    r'2.038981 + 0.238953 \cdot t + 0.003945 \cdot t^2 - 2.107753 \cdot e^{-t} + 0.038485 \cdot \sin(t) + 0.067269 \cdot \cos(t) + 9.593859 \cdot t \cdot e^{-t}, & \quad t \in [0, 10) \\ '
    r'5.515729 + 0.003311 \cdot t - 0.000008 \cdot t^2 - 1755.675474 \cdot t \cdot e^{-t}, & \quad t \in [10, 600) \\ '
    r'4.159792 + 0.000892 \cdot t, & \quad t \in [600, 2000] '
    r'\end{array} \right.$',

    r'$y_3(t) = \left\{ \begin{array}{ll} '
    r'1.547157 - 1.243739 \cdot e^{-t} + 0.704968 \cdot t \cdot e^{-t}, & \quad t \in [0, 10) \\ '
    r'1.285488 + 0.000662 \cdot t + 10025.241525 \cdot e^{-t} + 0.000059 \cdot \cos(t) - 866.120766 \cdot t \cdot e^{-t}, & \quad t \in [10, 600) \\ '
    r'1.407959, & \quad t \in [600, 2000] '
    r'\end{array} \right.$'
]

# Add equations to the bottom right quadrant with modifications
equation_text = r'\textbf{\Large Functions Identified using SINDy:}' + "\n\n"
features = [
    r'\normalsize Feature: \textit{\normalsize Velocity of Fluid at Output 1}',
    r'\normalsize Feature: \textit{\normalsize Mass Flow Rate of Gas at Output 1}',
    r'\normalsize Feature: \textit{\normalsize Velocity of Gas at Output 2}'
]

for i, eq in enumerate(piecewise_functions):
    equation_text += f"{features[i]}:\n\n{r'\small ' + eq}\n\n\n\n"

# Update the text position and font sizes
axs[1, 1].text(0.5, 1.0, equation_text, verticalalignment='top', horizontalalignment='center', transform=axs[1, 1].transAxes)
axs[1, 1].axis('off')

plt.tight_layout()
plt.show()