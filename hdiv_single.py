from firedrake import *

#ai = Constant(1+1j)
ai = Constant(1)

R = 6371220.
H = Constant(5960.)

mesh = IcosahedralSphereMesh(radius=R, refinement_level=1, degree=3)
x = SpatialCoordinate(mesh)
mesh.init_cell_orientations(x)

x, y, z = SpatialCoordinate(mesh)

outward_normals = CellNormal(mesh)
perp = lambda u: cross(outward_normals, u)


V1 = FunctionSpace(mesh, "BDM", 2)

u = TrialFunction(V1)
v = TestFunction(V1)

Omega = Constant(7.292e-5)  # rotation rate
R = Constant(R)
f = 2*Omega*z/R  # Coriolis parameter
g = Constant(9.8)  # Gravitational constant
tau = Constant(60*60)

a = (inner(ai*u,v) + tau**2*g*H*inner(div(u)/ai,div(v))
      - tau*inner(f*perp(u),v))*dx

import math
f1 = as_vector([0,exp((x+y+z)/R)*x*y*z/R**3,0])
F = inner(f1,v)*dx

params = {
    "ksp_type": "gmres",
    "ksp_rtol": 1e-8,
    "pc_type": "mg",
    "pc_mg_type": "full",
    "mg_levels_ksp_type": "richardson",
    "mg_levels_ksp_norm_type": "unpreconditioned",
    "mg_levels_ksp_monitor_true_residualx": None,
    "mg_levels_ksp_richardson_scale": 0.5,
    "mg_levels_pc_type": "python",
    "mg_levels_pc_python_type": "firedrake.PatchPC",
    "mg_levels_patch_pc_patch_save_operators": True,
    "mg_levels_patch_pc_patch_partition_of_unity": False,
    "mg_levels_patch_pc_patch_construct_type": "star",
    "mg_levels_patch_pc_patch_construct_dim": 0,
    "mg_levels_patch_pc_patch_sub_mat_type": "seqdense",
    "mg_levels_patch_sub_ksp_type": "preonly",
    "mg_levels_patch_sub_pc_type": "lu",
    "mg_coarse_pc_type": "lu",
    "ksp_monitor": True
}

w = Function(W)

Prob = LinearVariationalProblem(a, F, w, aP=aP)
Solver = LinearVariationalSolver(Prob, solver_parameters=params)

Solver.solve()
