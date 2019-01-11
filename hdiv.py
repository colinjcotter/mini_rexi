from firedrake import *

#ai = Constant(1+1j)
ai = Constant(1)

R = 6371220.
H = Constant(5960.)

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
R = Constant(R)
f = 2*Omega*z/R  # Coriolis parameter
g = Constant(9.8)  # Gravitational constant
tau = Constant(60*60)

a = (
    inner(ai*u,v) - tau*inner(f*perp(u),v) + 
    tau*g*inner(h,div(v))
    +inner(ai*h,phi) - tau*H*inner(div(u),phi)
)*dx

aP = (inner(ai*u,v) + tau**2*g*H*inner(div(u)/ai,div(v))
      - tau*inner(f*perp(u),v)
      + inner(ai*h,phi))*dx

import math
f1 = exp((x+y+z)/R)*x*y*z/R**3
F = inner(f1,phi)*dx

params = {
    "ksp_type": "gmres",
    "ksp_rtol": 1e-8,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_0_fields":"1",
    "pc_fieldsplit_1_fields":"0",
    "pc_fieldsplit_schur_fact_type": "full",
    "pc_fieldsplit_off_diag_use_amat": True,
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "bjacobi",
    "fieldsplit_0_pc_sub_type": "ilu",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "lu",
    "fieldsplit_1_pc_factor_mat_solver_type": "mumps",
    "ksp_monitor": True
}

w = Function(W)

Prob = LinearVariationalProblem(a, F, w, aP=aP)
Solver = LinearVariationalSolver(Prob, solver_parameters=params)

Solver.solve()
