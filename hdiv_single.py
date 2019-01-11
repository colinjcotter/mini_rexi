from firedrake import *

ai = Constant(1)

n = 16
mesh = PeriodicUnitSquareMesh(n, n)
V1 = FunctionSpace(mesh, "BDM", 2)

u = TrialFunction(V1)
v = TestFunction(V1)

f = Constant(1.0e-12)
a = (ai*inner(u,v) + inner(div(u)/ai,div(v))
      - inner(f*perp(u),v))*dx

x, y = SpatialCoordinate(mesh)
import math
f1 = as_vector([sin(2*math.pi*x)*sin(2*math.pi*y)*exp(-10*(x**2 + y**2)),0])

F = inner(f1,v)*dx

params = {
    "ksp_type": "gmres",
    "ksp_rtol": 1e-8,
    "pc_type": "python",
    "ksp_monitor": True,
    "pc_python_type": "firedrake.PatchPC",
    "patch_pc_patch_save_operators": True,
    "patch_pc_patch_partition_of_unity": False,
    "patch_pc_patch_construct_type": "vanka",
    "patch_pc_patch_construct_dim": 0,
    "patch_pc_patch_sub_mat_type": "seqdense",
    "patch_sub_ksp_type": "preonly",
    "patch_sub_pc_type": "lu",
}

w = Function(V1)

Prob = LinearVariationalProblem(a, F, w)
Solver = LinearVariationalSolver(Prob, solver_parameters=params)

Solver.solve()
