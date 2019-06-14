from firedrake import *

R = 1.0

base = IcosahedralSphereMesh(radius=R, refinement_level=0)
nref = 5
mh = MeshHierarchy(base, nref)
for mesh in mh:
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)    

mesh = mh[-1]

x, y, z = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V)
v = TestFunction(V)

a = (u*v + inner(u,v))*dx


f1 = exp((x+y+z)/R)*x*y*z/R**3
F = v*f1*dx

lu_params = {
    "ksp_monitor": None,
    "ksp_type": "gmres",
    "pc_type": "lu",
    "mat_type": 'aij'
}

mg_params = {"mat_type": "matfree",
             "snes_type": "ksponly",
             "ksp_type": "gmres",
             "ksp_rtol": 1.0e-8,
             "ksp_atol": 0.0,
             "ksp_max_it": 1000,
             "ksp_monitor": None,
             "ksp_converged_reason": None,
             "ksp_norm_type": "unpreconditioned",
             "pc_type": "mg",
             "mg_coarse_ksp_type": "preonly",
             "mg_coarse_pc_type": "python",
             "mg_coarse_pc_python_type": "firedrake.AssembledPC",
             "mg_coarse_assembled_pc_type": "lu",
             "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
             "mg_levels_ksp_type": "richardson",
             "mg_levels_ksp_max_it": 1,
             "mg_levels_pc_type": "bjacobi",
             "mg_levels_sub_pc_type": "ilu"}

w = Function(V)

mg = False
if mg:
    Prob = LinearVariationalProblem(a, F, w)
    Solver = LinearVariationalSolver(Prob,
                                     solver_parameters=mg_params)
else:
    Prob = LinearVariationalProblem(a, F, w)
    Solver = LinearVariationalSolver(Prob, solver_parameters=lu_params)
Solver.solve()

f0 = File('block.pvd')
f0.write(w)

