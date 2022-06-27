import numpy as np
import matplotlib.pyplot as plt

class OptimizationProblem:
    """Base Optimization Problem"""
    def __init__(self, max_iterations=50, learning_rate=0.002, tolerance=10 ** -4):
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.tolerance = tolerance

        self.costFunction = None
        self.gradFunction = None

        self.Xpath = []
        self.Ypath = []
        self.Zpath = []

    def evaluate_Jacobian(self, x, y):
        """Calculate the Gradient at the given point.

        Keyword arguments:
            arg_1 (float): current x location
            arg_2 (float): current y location

        Returns:
            dx (float): partial x derivative calculated at the input location
            dy (float): partial y derivative calculated at the input location

        Additional Information:
            When m=1 the Jacobian is the same as the Gradient,
            since it is a generalization of the Gradient.
        """
        if self.gradFunction == None:
            print("You must define a gradient function")
            return -1
        else:
            dx, dy = self.gradFunction(x, y)
            return dx, dy

    def solve(self, x, y):
        """Perform Gradient Descent Algorithm starting at the input location.
        Print out results and stopping condition.

        Keyword arguments:
            arg_1 (float): starting x value for gradient descent
            arg_2 (float): starting y value for gradient descent

        Returns:
            None

        Errors:
            Returns -1 if there is no cost function or gradient function defined for the problem to solve
        """
        if self.costFunction == None or self.gradFunction == None:
            print("You must define a cost function and gradient function")
            return -1

        flag = 0

        for iteration in range(self.max_iterations):
            previousEval = self.costFunction(x, y)
            dx, dy = self.evaluate_Jacobian(x, y)
            x = x - self.learning_rate * dx
            y = y - self.learning_rate * dy
            newEval = self.costFunction(x, y)
            residual = abs(previousEval - newEval)

            print(f"Iteration {iteration}")
            print(f"The location now is x = {x}, y = {y}")
            print(f"The value of the cost function is {newEval}")
            print(f"The residual is: {residual}")

            self.Xpath.append(x)
            self.Ypath.append(y)
            self.Zpath.append(newEval)

            if residual < self.tolerance:
                flag = 1
                break
        if flag:
            print("Tolerance Achieved")
        else:
            print("Maximum Number Iterations Hit Without Convergence")

    def visualize(self, x, y, size=10, resolution=100):
        """Plot the Gradient Descent Path and Surrounding Cost Function.

        Keyword arguments:
            arg_1 (float): starting x value for gradient descent
            arg_2 (float): starting y value for gradient descent
            arg_3 (float, optional): how far from the initial point in both directions will be visualized
            arg_4 (float, optional): granularity of the visualization.  Higher value--> better quality

        Returns:
            None
        """
        if self.Xpath == []:
            print("You Must Solve The Gradient Descent Problem First")
        if self.costFunction == None or self.gradFunction == None:
            print("You must define a cost function and gradient function")

        d1 = np.linspace(x - size, x + size, resolution)
        d2 = np.linspace(y - size, y + size, resolution)
        costX, costY = np.meshgrid(d1, d2)
        costZ = self.costFunction(costX, costY)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(costX, costY, costZ, cmap='Oranges', edgecolor=None, alpha=0.5)
        path = ax.scatter3D(self.Xpath, self.Ypath, self.Zpath, c=(self.Zpath), cmap='viridis',
                            label="Gradient Descent Path")
        ax.scatter3D(self.Xpath[0], self.Ypath[0], self.Zpath[0], c="Black", label="Start")
        ax.scatter3D(self.Xpath[-1], self.Ypath[-1], self.Zpath[-1], c="Red", label="End")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Gradient Descent")
        plt.colorbar(path)
        plt.legend()
        plt.show()


class GDOptProblem(OptimizationProblem):
    """This extends the OptimizationProblem class.  It uses Gradient Descent to minimize a specific cost function """

    def __init__(self, max_iterations=50, learning_rate=0.01, tolerance=10 ** -4):
        super().__init__(max_iterations, learning_rate, tolerance)
        self.costFunction = lambda x, y: (x ** 2 + y - 3) ** 2 + (x + y ** 2 - 9) ** 2
        self.gradFunction = lambda x, y: (4 * x ** 3 + 4 * x * y - 10 * x + 2 * y ** 2 - 18, 4 * y ** 3 + 4 * x * y - 34 * y + 2 * x ** 2 - 6)

