#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

mpl.rcParams["font.size"] = 20

class node:
  def __init__(self, id, coords):
    self.id_ = id
    self.coords_ = coords

  def Id(self):
    return self.id_

  def Coordinates(self):
    return self.coords_

  def X(self):
    return self.coords_[0]

  def Y(self):
    return self.coords_[1]

  def Z(self):
    return self.coords_[2]

  def Distance(self, a):
    return np.linalg.norm(self.coords_, a.coords_)

  def Print(self):
    print("id:", self.id_, "coordinates:", self.coords_)

class triangle:
  def __init__(self, id, n0, n1, n2):
    self.id_ = id
    self.connectivity_ = np.array([n0.Id(), n1.Id(), n2.Id()])
    self.r01_ = n1.Coordinates() - n0.Coordinates()
    self.r12_ = n2.Coordinates() - n1.Coordinates()
    self.r20_ = n0.Coordinates() - n2.Coordinates()
    area = 0.5 * np.cross(-self.r20_, self.r01_)
    self.area_ = np.linalg.norm(area)
    self.normal_ = area / self.area_
    self.centroid_ = (n0.Coordinates() + n1.Coordinates() + n2.Coordinates()) / 3
    self.shape_ = np.zeros((3, 3))
    self.shapeDeriv_ = np.zeros((3, 2))
    self.CalculateShapeFunctions(n0, n1, n2)
    self.weight_ = np.zeros((3, 3))
    self.weightDeriv_ = np.zeros((3, 2))
    self.CalculateWeightFunctions()
    self.massMatrix_ = np.zeros((3, 3))
    self.CalculateMassMatrix()
    self.boundaryEdge_ = [False, False, False]

  # calculate shape/trial functions
  def CalculateShapeFunctions(self, n0, n1, n2):
    self.shape_[0, 0] = n1.X() * n2.Y() - n2.X() * n1.Y()
    self.shape_[1, 0] = n2.X() * n0.Y() - n0.X() * n2.Y()
    self.shape_[2, 0] = n0.X() * n1.Y() - n1.X() * n0.Y()
    self.shape_[0, 1] = n1.Y() - n2.Y()
    self.shape_[1, 1] = n2.Y() - n0.Y()
    self.shape_[2, 1] = n0.Y() - n1.Y()
    self.shape_[0, 2] = n2.X() - n1.X()
    self.shape_[1, 2] = n0.X() - n2.X()
    self.shape_[2, 2] = n1.X() - n0.X()
    self.shape_ *= 0.5 / self.area_
    self.shapeDeriv_ = np.zeros_like(self.shape_)
    self.shapeDeriv_[:, 1:] = self.shape_[:, 1:]
    
  # calculate weight/test functions
  # same as shape/trial functions for Galerkin method
  def CalculateWeightFunctions(self):
    self.weight_ = self.shape_.copy()
    self.weightDeriv_ = self.shapeDeriv_.copy()

  # Shape vector times its transpose integrated over the element area
  def CalculateMassMatrix(self):
    self.massMatrix_ = np.ones((3, 3))
    self.massMatrix_[0, 0] = 2
    self.massMatrix_[1, 1] = 2
    self.massMatrix_[2, 2] = 2
    self.massMatrix_ *= self.area_ / 12

  def SetEdgeAsBoundary(self, ind):
    self.boundaryEdge_[ind] = True

  def Id(self):
    return self.id_

  def Connectivity(self):
    return self.connectivity_

  def MatrixIndices(self):
    indices = []
    for r in self.connectivity_:
      for c in self.connectivity_:
        indices.append((r, c))
    return indices

  def VectorIndicesEdge(self, ind):
    edgeInds = []
    if ind == 0:
      edgeInds.append(self.connectivity_[0])
      edgeInds.append(self.connectivity_[1])
    elif ind == 1:
      edgeInds.append(self.connectivity_[1])
      edgeInds.append(self.connectivity_[2])
    elif ind == 2:
      edgeInds.append(self.connectivity_[0])
      edgeInds.append(self.connectivity_[2])
    return edgeInds

  def MassMatrix(self):
    return self.massMatrix_

  def Area(self):
    return self.area_

  def Normal(self):
    return self.normal_

  def ShapeInterp(self, nd):
    vec = np.ones((3, 1))
    vec[1] = nd.X()
    vec[2] = nd.Y()
    return np.sum(self.shape_ @ vec)

  def ShapeDerivative(self):
    return self.shapeDeriv_

  def WeightDerivative(self):
    return self.weightDeriv_

  def Print(self):
    print("id:", self.id_, "connectivity:", self.connectivity_)

def AssembleMatrices(numNodes, elems):
  K = np.zeros((numNodes, numNodes))
  M = np.zeros((numNodes, numNodes))
  for elem in elems:
    elemK = elem.WeightDerivative() @ np.transpose(elem.ShapeDerivative())
    elemK *= elem.Area()
    elemK = elemK.flatten("C")
    elemM = elem.MassMatrix()
    elemM = elemM.flatten("C")
    indices = elem.MatrixIndices()
    for ii in range(len(indices)):
      K[indices[ii]] += elemK[ii]
      M[indices[ii]] += elemM[ii]
  return K, M

def AssignNeumannBCs(rhs, elems, qdot, inds):
  edgeFactor = 0.5  # 2 nodes per edge
  for elem in elems:
    if elem.boundaryEdge_[0]:
      length = np.linalg.norm(elem.r01_)
      indices = elem.VectorIndicesEdge(0)
      for ii in indices:
        if ii in inds:
          rhs[ii] += edgeFactor * qdot * length
    if elem.boundaryEdge_[1]:
      length = np.linalg.norm(elem.r12_)
      indices = elem.VectorIndicesEdge(1)
      for ii in indices:
        if ii in inds:
          rhs[ii] += edgeFactor * qdot * length
    if elem.boundaryEdge_[2]:
      length = np.linalg.norm(elem.r20_)
      indices = elem.VectorIndicesEdge(2)
      for ii in indices:
        if ii in inds:
          rhs[ii] += edgeFactor * qdot * length
  return rhs

def AssignDirichletBCs(Morig, rhs, inds):
  M = Morig.copy()
  # assigned dT for all Dirichlet BCs
  dT = 0.0
  for jj in inds:
    for ii in range(M.shape[0]):
      rhs[ii] -= M[ii, jj] * dT
      M[:, jj] = 0.0
      M[jj, :] = 0.0
      M[jj, jj] = 1.0
      rhs[jj] = dT
  return M, rhs