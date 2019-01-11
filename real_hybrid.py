from firedrake import *

n = 50
R = 6371220.
H = 5960.

mesh = IcosahedralSphereMesh(radius=R, refinement_level=3, degree=3)
x = SpatialCoordinate(mesh)
mesh.init_cell_orientations(x)

x, y, z = SpatialCoordinate(mesh)

outward_normals = CellNormal(mesh)
perp = lambda u: cross(outward_normals, u)
    
V1 = FunctionSpace(mesh, "BDM", 2)
V2 = FunctionSpace(mesh, "DG", 1)
W = MixedFunctionSpace((V1, V2))

u, h = TrialFunctions(W)
v, phi = TestFunctions(W)

Omega = Constant(7.292e-5)  # rotation rate
f = 2*Omega*z/R  # Coriolis parameter
g = Constant(9.8)  # Gravitational constant
tau = Constant(60*60*6)

a = (
    inner(u,v)*dx - inner(f*tau*perp(u),v)*dx
    +tau*g*h*div(v)*dx
    +h*phi*dx - H*tau*div(u)*phi*dx
)

f1 = exp((x+y+z)/R)*x*y*z/R**3

F = f1*phi*dx

params = {
    'mat_type': 'matfree',
    'ksp_monitor': True,
    'ksp_monitor_true_residual': True,
    'ksp_type': 'fgmres',
    'pc_type': 'python',
    'pc_python_type': 'firedrake.HybridizationPC',
    'hybridization': {'ksp_type': 'gmres',
                      'ksp_monitor':True,
                      'ksp_converged_reason':True,
                      'pc_type': 'gamg'}}

w = Function(W)

Prob = LinearVariationalProblem(a, F, w)
Solver = LinearVariationalSolver(Prob, solver_parameters=params)

Solver.solve()
