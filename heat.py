#!/usr/bin/python3

import fem
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.size"] = 20

def AddEdgesToMap(elem, edgeMap, edgeCount):
  ec = elem.EdgeConnectivity()
  for ii in range(ec.shape[0]):
    candidateEdge = (ec[ii,0], ec[ii,1])
    if ec[ii,0] > ec[ii,1]:
      candidateEdge = (ec[ii,1], ec[ii,0])
    if candidateEdge not in edgeMap:
      edgeMap[candidateEdge] = edgeCount
      edgeCount += 1
  return edgeMap, edgeCount

def DefineGrid(xx, yy):
  nodes = []
  id = 0
  for x in xx:
    for y in yy:
      coords = np.array([x, y])
      nodes.append(fem.node(id, coords))
      id += 1
  elems = []
  edgeMap = {}
  edgeCount = 0
  id = 0
  for x in range(len(xx) - 1):
    for y in range(len(yy) - 1):
      bl = x * len(xx) + y
      tl = x * len(xx) + y + 1
      br = (x + 1) * len(xx) + y
      tr = (x + 1) * len(xx) + y + 1
      # determine if edge is on boundary
      leftBC = False
      if x == 0:
        leftBC = True
      rightBC = False
      if x == len(xx) - 2:
        rightBC = True
      bottomBC = False
      if y == 0:
        bottomBC = True
      topBC = False
      if y == len(yy) - 2:
        topBC = True

      elems.append(fem.triangle(id, nodes[bl], nodes[tl], nodes[tr]))
      edgeMap, edgeCount = AddEdgesToMap(elems[-1], edgeMap, edgeCount)
      if leftBC:
        elems[id].SetEdgeAsBoundary(0)
      if topBC:
        elems[id].SetEdgeAsBoundary(1)
      id += 1
      elems.append(fem.triangle(id, nodes[bl], nodes[tr], nodes[br]))
      edgeMap, edgeCount = AddEdgesToMap(elems[-1], edgeMap, edgeCount)
      if rightBC:
        elems[id].SetEdgeAsBoundary(1)
      if bottomBC:
        elems[id].SetEdgeAsBoundary(2)
      id += 1
    edges = np.zeros((edgeCount, 2), dtype=int)
    for connectivity, eid in edgeMap.items():
      edges[eid,:] = [connectivity[0], connectivity[1]]
    elemsByEdge = np.zeros((len(elems), 3), dtype=int)
    for jj in range(len(elems)):
      ec = elems[jj].EdgeConnectivity()
      for ii in range(ec.shape[0]):
        if (ec[ii, 0], ec[ii, 1]) in edgeMap:
          elemsByEdge[jj, ii] = edgeMap[(ec[ii, 0], ec[ii, 1])]
        elif (ec[ii, 1], ec[ii, 0]) in edgeMap:
          elemsByEdge[jj, ii] = edgeMap[(ec[ii, 1], ec[ii, 0])]
        else:
          print("ERROR: Cant find edge in element list", ec[ii])
  return nodes, elems, edges, elemsByEdge

def ReshapeCoords(nodes, numI, numJ):
  x = np.zeros((len(nodes), 1))
  y = np.zeros((len(nodes), 1))
  ii = 0
  for node in nodes:
    x[ii] = node.X()
    y[ii] = node.Y()
    ii += 1
  x = np.reshape(x, (numI, numJ), order='C')
  y = np.reshape(y, (numI, numJ), order='C')
  return x, y
    
def StableTimestep(elems, nu):
  dt = 1e30
  for elem in elems:
    # factor 2 for 2d, factor 2 for triangle
    dt = min(dt, 0.125 * elem.Area() / nu)
  return dt

def main():
  # Set up options
  parser = argparse.ArgumentParser()
  parser.add_argument("-k", "--conductivity", action="store",
                     dest="conductivity", default=1.0, type=float,
                     help="thermal conductivity, k. Default = 1.0")
  parser.add_argument("-c", "--specificHeat", action="store",
                     dest="specificHeat", default=1.0, type=float,
                     help="specific heat, cp. Default = 1.0")
  parser.add_argument("-d", "--density", action="store",
                     dest="density", default=1.0, type=float,
                     help="material density, rho. Default = 1.0")
  parser.add_argument("-i", "--initial", action="store", dest="initial",
                     default=100.0, type=float,
                     help="initial temperature. Default = 100.0")
  parser.add_argument("-n", "--timeSteps", action="store", dest="timeSteps",
                     default=1, type=int,
                     help="number of time steps. Default = 1")
  parser.add_argument("-s", "--size", action="store", dest="size",
                     default=3, type=int,
                     help="number of nodes per side. Default = 3")
  parser.add_argument("-l", "--left", action="store", dest="left",
                     default=500.0, type=float,
                     help="left edge temperature. Default = 500.0")
  parser.add_argument("-r", "--right", action="store", dest="right",
                     default=1.0, type=float,
                     help="right edge heat flux. Default = 1.0")

  args = parser.parse_args()

  # ------------------------------------------------------------------
  # baseline simulation
  print("-----------------------------------------")
  print("Baseline Simulation")
  print("-----------------------------------------")
  numPerSide = args.size
  xx = np.linspace(0, 1, numPerSide)
  yy = np.linspace(0, 1, numPerSide)
  nodes, elems, edges, elemsByEdge = DefineGrid(xx, yy)
  x, y = ReshapeCoords(nodes, len(xx), len(yy))
  for node in nodes:
    node.Print()
  for elem in elems:
    elem.Print()
  for eid in range(edges.shape[0]):
    print("Edge id {0} connects nodes {1} and {2}".format(eid, edges[eid, 0], \
      edges[eid, 1]))
  for eid in range(elemsByEdge.shape[0]):
    print("Element {0} has edges {1}, {2}, and {3}".format(eid, \
      elemsByEdge[eid,0], elemsByEdge[eid,1], elemsByEdge[eid,2]))
  numNodes = len(nodes)

  # initialize temperature
  T = np.ones((len(nodes), 1))
  T *= args.initial

  # left boundary temperature
  tl = args.left
  T[:numPerSide] = tl

  # right boundary heat flux
  qr = args.right

  # assemble matrices
  K, M = fem.AssembleMatrices(len(nodes), elems)
  
  # caclulate time step
  nu = args.conductivity / (args.specificHeat * args.density)
  dt = StableTimestep(elems, nu)

  # calculate source term
  S = np.zeros_like(T)

  # calculate right hand side
  rhs = nu * (-K @ T + S)

  # assign boundary conditions and calculate reduced mass matrix
  dirichletInds = list(range(numPerSide))
  neumannInds = list(range(numNodes - numPerSide, numNodes))
  print("Dirichlet Indices", dirichletInds)
  print("Neumann Indices", neumannInds)
  print("Initial Conditions\n", T, "\n")
  Mmod, rhs = fem.AssignDirichletBCs(M, rhs, dirichletInds)
  Minv = np.linalg.inv(Mmod)

  for _ in range(args.timeSteps):

    # calculate right hand side
    rhs = nu * (-K @ T + S)

    # assign boundary conditions
    rhs = fem.AssignNeumannBCs(rhs, elems, qr, neumannInds)
    _, rhs = fem.AssignDirichletBCs(M, rhs, dirichletInds)
  
    # solve MT,t = -KT+S
    T += dt * Minv @ rhs
  
  print("Final Solution")
  print(T)
  
  # ------------------------------------------------------------------
  # plot solutions
  _, ax = plt.subplots(1, 1, figsize=(16, 8))
  ax.set_xlabel("X (m)")
  ax.set_ylabel("Y (m)")
  ax.set_title("Temperature Contour")
  cf=ax.contourf(x, y, np.reshape(T, (len(xx), len(yy)), order='C'), \
                levels=np.linspace(np.min(T), np.max(T), 11))
  cbar=plt.colorbar(cf, ax=ax)
  cbar.ax.set_ylabel("Temperature (K)")
  ax.grid(True)
  plt.show()


if __name__ == "__main__":
  main()
