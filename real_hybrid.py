from firedrake import *

n = 32
mesh = PeriodicUnitSquareMesh(n, n)
V1 = FunctionSpace(mesh, "BDM", 2)
V2 = FunctionSpace(mesh, "DG", 1)
W = MixedFunctionSpace((V1, V2))

u, h = TrialFunctions(W)
v, phi = TestFunctions(W)

f = Constant(1.0e-4)  # Coriolis parameter
g = Constant(9.8)  # Gravitational constant
H = Constant(1000.)  # Mean depth
tau = Constant(10.)

a = (
    inner(u,v) - inner(f*tau*perp(u),v) + 
    inner(h,div(v))
    +inner(h,phi) - H*inner(tau*div(u),phi)
)*dx

x, y = SpatialCoordinate(mesh)
import math
f1 = sin(2*math.pi*x)*sin(2*math.pi*y)*exp(-10*(x**2 + y**2))

F = inner(f1,phi)*dx

params = {
    'ksp_type': 'preonly',
    'pc_type': 'python',
    'pc_python_type': 'firedrake.HybridizationPC',
    'hybridization': {'ksp_type': 'preonly',
                      'pc_type': 'lu'}}

w = Function(W)

Prob = LinearVariationalProblem(a, F, w)
Solver = LinearVariationalSolver(Prob, solver_parameters=params)

Solver.solve()
