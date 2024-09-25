import numpy as np
import matplotlib.pyplot as plt
import pyDOE
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
class PSO:
    def __init__(self, MNIt, NOP, NV, MaxB, MinB, InertiaMin, InertiaMax, C1I, C1F, C2I, C2F, exploitation_factor=1.5):
        self.MNIt = MNIt  # Maximum number of iterations
        self.NOP = NOP   # Number of particles
        self.NV = NV     # Number of variables
        self.MaxB = MaxB  # Maximum boundary
        self.MinB = MinB  # Minimum boundary
        self.InertiaMin = InertiaMin  # min inertia weight is for exploration. Decrease results in more exploitation.
        self.InertiaMax = InertiaMax  # max inertia weight is for exploitation. Increase results in more exploration.
        self.C1I = C1I  # cognitive component at the beginning
        self.C1F = C1F  # cognitive component at the end
        self.C2I = C2I  # social component at the beginning
        self.C2F = C2F  # social component at the end
        self.exploitation_factor = exploitation_factor  # exploitation factor for mentor and independent roles

    def update_InertiaWeight(self, It):
        return self.InertiaMax - (It / self.MNIt) * (self.InertiaMax - self.InertiaMin)

    def define_role(self, LocalBestScore, i):
        if LocalBestScore[i] < np.percentile(LocalBestScore, 25):
            return "mentor"
        elif LocalBestScore[i] > np.percentile(LocalBestScore, 75):
            return "mentee"
        else:
            return "independent"


    
    def update_C1C2(self, It):
        # Sigmoidal parameters
        alpha = 10  # Controls the steepness of the sigmoid curve
        beta = self.MNIt / 2  # Controls the midpoint of the sigmoid curve

        # Update cognitive component (C1) with a decreasing sigmoidal function
        # Starts high for exploration, then decreases to favor exploitation
        C1 = self.C1I + (self.C1F - self.C1I) * (1 - sigmoid(alpha * (It - beta) / self.MNIt))

        # Update social component (C2) with an increasing sigmoidal function
        # Starts low, then increases to favor more collective behavior
        C2 = self.C2I + (self.C2F - self.C2I) * sigmoid(alpha * (It - beta) / self.MNIt)

        return C1, C2


   # def objfun(self, x, A=10):
   #     n = len(x)
   #     return A * n + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)
    # write a complex multi modal 
    def objfun(self, x):
        #return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    def optimize(self):
        # Generate initial positions using Latin Hypercube Sampling
        X_initial = pyDOE.lhs(self.NV, samples=self.NOP, criterion='maximin')
        X_initial = self.MinB + (self.MaxB - self.MinB) * X_initial  # Scale to problem bounds

        # Evaluate initial positions
        ## in parallel processing using multiprocessing
        ## this is a prototype of parallel processing and not necessary suitable for this objective function
        
        from multiprocessing import Process
        
        processes = []
        
        for x in X_initial:
            process = Process(target=self.objfun, args=(x,))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
        
        scores_initial = np.array([self.objfun(x) for x in X_initial])

        # Select best positions as starting points
        X = np.array([X_initial[i] for i in np.argsort(scores_initial)[:self.NOP]])
        V = np.zeros((self.NOP, self.NV))  # Initial velocities
        Vmax = 0.1 * (self.MaxB - self.MinB)

        # Initialize local and global bests
        for i in range(self.NOP):
            LocalBest = np.copy(X)
            
            LocalBestScore = np.array([self.objfun(x) for x in LocalBest])
            GlobalBest = X[np.argmin(LocalBestScore)]
            GlobalBestScore = np.min(LocalBestScore)
            GlobalBestScore_collection = [GlobalBestScore]

        # PSO main loop
        for It in range(self.MNIt):
            print("Iteration", It, "Best Score:", GlobalBestScore)
            # Update inertia weight linearly
            InertiaWeight = self.update_InertiaWeight(It)

            # Update cognitive and social components linearly
            C1, C2 = self.update_C1C2(It)

            for i in range(self.NOP):
                # Update velocities
                R1 = np.random.uniform(0, 1, self.NV)
                R2 = np.random.uniform(0, 1, self.NV)

                role = self.define_role(LocalBestScore, i)

                if role == "mentee":
                    # this randomly turns off the social and cognitive components
                    Cse, Sme = (1, 0) if np.random.randint(0, 2) == 1 else (0, 1)

                    V[i] = (InertiaWeight * V[i] +
                            C1 * R1 * (GlobalBest - X[i]) * Cse +
                            C2 * R2 * (LocalBest[i] - X[i]) * Sme)

                elif role == "mentor":
                    # Increase exploitation: More weight to best known positions
                    V[i] = (InertiaWeight * V[i] +
                            C1 * R1 * (LocalBest[i] - X[i]) +
                            self.exploitation_factor * C2 * R2 * (GlobalBest - X[i]))

                elif role == "independent":
                    # Balanced search
                    V[i] = (InertiaWeight * V[i] +
                            self.exploitation_factor * C1 * R1 * (LocalBest[i] - X[i]) +
                            C2 * R2 * (GlobalBest - X[i]))

                # Apply velocity limits
                V[i] = np.clip(V[i], -Vmax, Vmax)
                X[i] += V[i]

                # Enhanced velocity correction
                for j in range(len(X[i])):
                    if X[i][j] < self.MinB[j]:
                        X[i][j] = self.MinB[j]
                        # randomly reverse the velocity or make it zero
                        V[i][j] = 0.5 * V[i][j] if np.random.randint(0, 2) == 1 else 0
                    elif X[i][j] > self.MaxB[j]:
                        X[i][j] = self.MaxB[j]
                        V[i][j] = 0.5 * V[i][j] if np.random.randint(0, 2) == 1 else 0

                # Evaluate the new position
                score = self.objfun(X[i])

                # Update local best
                if score < LocalBestScore[i]:
                    LocalBest[i] = X[i]
                    LocalBestScore[i] = score

                # Update global best
                if score < GlobalBestScore:
                    GlobalBest = X[i]
                    GlobalBestScore = score

            GlobalBestScore_collection.append(GlobalBestScore)

        plt.figure(figsize=(10, 6))
        plt.plot(GlobalBestScore_collection, color='b', marker='o', linestyle='-', linewidth=2, markersize=6)
        plt.xlabel('Iterations')
        plt.ylabel('Objective Value')
        plt.title('Global Best Improvement')
        plt.grid(True, which="both", ls="--", c='gray', alpha=0.5)
        plt.show()


        # plot the progress of the global best score
        #plt.plot(GlobalBestScore_collection)
       # plt.show()

        # Output the best solution
        print("Best Position:", GlobalBest)
        print("Best Score:", GlobalBestScore)

# Parameters
MNIt = 100  # Maximum number of iterations
NOP = 100   # Number of particles
NV = 2      # Number of variables
MaxB = np.array([1000, 1000])  # Maximum boundary
MinB = np.array([0, 0])  # Minimum boundary

# PSO coefficients
InertiaMin = 0.4
InertiaMax = 0.9
C1I = 1.5
C1F = 0.5
C2I = 0.5
C2F = 1.5

# Create MMPSO instance and optimize
pso = PSO(MNIt, NOP, NV, MaxB, MinB, InertiaMin, InertiaMax, C1I, C1F, C2I, C2F)
pso.optimize()
