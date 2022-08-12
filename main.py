from builtins import min

import scipy.spatial as sp
import pandas as pd
import numpy as np
import logging as log
import json
import sys

# also required packages: openpyxl
from ortools.linear_solver import pywraplp


def solve_OrTools(dima):
    """
    generate mip model using google or-tools and solve it

    :param dima: the distance matrix
    :return:  solution X, model, status
    """

    if dima.ndim != 2 or dima.shape[0] != dima.shape[1]:
        raise ValueError("Invalid dima dimensions detected. Square matrix expected.")

    # determine number of nodes
    num_nodes = dima.shape[0]
    all_nodes = range(0, num_nodes)
    all_but_first_nodes = range(1, num_nodes)

    # Create the model.
    solver_name = "CBC_MIP"
    log.info("Instantiating solver " + solver_name)
    model = pywraplp.Solver.CreateSolver(solver_name)
    model.EnableOutput()
    model.SetTimeLimit(200 * 1000)

    log.info("Defining MIP model... ")
    # generating decision variables X_ij
    log.info("Creating " + str(num_nodes * num_nodes) + " boolean x_ij variables... ")
    x = {}
    for i in all_nodes:
        for j in all_nodes:
            x[(i, j)] = model.BoolVar("x_i%ij%i" % (i, j))

    log.info("Creating " + str(num_nodes) + " boolean u_i variables... ")
    u = {}
    for i in all_nodes:
        u[i] = model.IntVar(0, num_nodes, "u_i%i" % i)

    # constraint 1: leave every point exactly once
    log.info("Creating " + str(num_nodes) + " Constraint 1... ")
    for i in all_nodes:
        model.Add(sum(x[(i, j)] for j in all_nodes) == 1)

    # constraint 2: reach every point from exactly one other point
    log.info("Creating " + str(num_nodes) + " Constraint 2... ")
    for j in all_nodes:
        model.Add(sum(x[(i, j)] for i in all_nodes) == 1)

    # constraint 3.1: subtour elimination constraints (Miller-Tucker-Zemlin) part 1
    log.info("Creating 1 Constraint 3.1... ")
    model.Add(u[0] == 1)

    # constraint 3.2: subtour elimination constraints (Miller-Tucker-Zemlin) part 2
    log.info("Creating " + str(len(all_but_first_nodes)) + " Constraint 3.2... ")
    for i in all_but_first_nodes:
        model.Add(2 <= u[i])
        model.Add(u[i] <= num_nodes)

    # constraint 3.3: subtour elimination constraints (Miller-Tucker-Zemlin) part 3
    log.info("Creating " + str(len(all_but_first_nodes)) + " Constraint 3.2... ")
    for i in all_but_first_nodes:
        for j in all_but_first_nodes:
            model.Add(u[i] - u[j] + x[(i, j)] <= (num_nodes - 1) * (1 - x[(i, j)]))

    # Minimize the total distance
    model.Minimize(sum(x[(i, j)] * dima[(i, j)] for i in all_nodes for j in all_nodes))

    log.info("Solving MIP model... ")
    status = model.Solve()
    # check problem response
    if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        log.info("Solution:")
        log.info("Objective value =" + str(model.Objective().Value()))
        sol = []
        sol_path2 = {}
        for i in all_nodes:
            for j in all_nodes:
                sol.append(x[(i, j)].solution_value())
                if x[(i, j)].solution_value() > 0:
                    sol_path2[i] = j
        sol = np.array(sol).reshape(num_nodes, num_nodes)
        print(np.sum(sol * dima))
        print(sol_path2)
        p = [sol_path2.keys()[0]]
        for i in sol_path2.values():
            p.append(sol_path2[i])
        print(p)
    elif status == pywraplp.Solver.INFEASIBLE:
        log.info("The problem is infeasible.")
    else:
        log.info("The problem could not be solved. Return state was: " + str(status))

    return u, model, status


def print_solution(u):
    num_nodes = len(u)
    all_nodes = range(0, num_nodes)
    path = []
    for i in all_nodes:
        path.append(str(int(u[i].solution_value())))
    return path
    # log.info("u(" + str(i) + ")=" + str(int(u[i].solution_value())))


def main():
    # configure logger for info level
    log.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=log.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    # load tsp instance
    tsp_problem = "dj38.tsp"

    log.info("Reading TSP problem Instance " + tsp_problem)
    tsp = pd.read_csv(
        tsp_problem,
        sep=" ",
        skiprows=10,
        dtype=float,
        names=["nodeId", "lat", "lng"],
        skipfooter=0,
        engine="python",
    )
    tsp = tsp.sort_values(by="nodeId", inplace=False)

    A = tsp[["lat", "lng"]].to_numpy()
    dima = sp.distance_matrix(A, A)

    # now solve problem
    solve_OrTools(dima)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main()
