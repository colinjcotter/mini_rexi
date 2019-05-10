from firedrake import *

R = 6371220.

nbase = 1
base = IcosahedralSphereMesh(radius=R, refinement_level=nbase)
nref = 5 - nbase

mh = MeshHierarchy(base, nref)
for mesh in mh:
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)    

mesh = mh[-1]

x, y, z = SpatialCoordinate(mesh)

outward_normals = CellNormal(mesh)
perp = lambda u: cross(outward_normals, u)


V1 = FunctionSpace(mesh, "BDM", 2)
V2 = FunctionSpace(mesh, "DG", 1)

ur = TrialFunction(V1)
vr = TestFunction(V1)

gamma = Constant(10.0)

a = (gamma*inner(ur, vr) + div(ur)*div(vr))*dx

f1 = exp((x+y+z)/R)*x*y*z/R**3
F = inner(div(vr),f1)*dx

lu_params = {
    "ksp_type": "preonly",
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
             "mg_levels_ksp_richardson_scale": 1,
             "mg_levels_pc_type": "python",
             "mg_levels_pc_python_type": "firedrake.PatchPC",
             "mg_levels_patch_pc_patch_save_operators": True,
             "mg_levels_patch_pc_patch_partition_of_unity": False,
             "mg_levels_patch_pc_patch_sub_mat_type": "seqaij",
             "mg_levels_patch_pc_patch_construct_type": "star",
             "mg_levels_patch_pc_patch_multiplicative": False,
             "mg_levels_patch_pc_patch_symmetrise_sweep": False,
             "mg_levels_patch_pc_patch_construct_dim": 0,
             "mg_levels_patch_sub_ksp_type": "preonly",
             "mg_levels_patch_sub_pc_type": "lu"}

patch_params = {"mat_type": "matfree",
             "snes_type": "ksponly",
             "ksp_type": "gmres",
             "ksp_rtol": 1.0e-8,
             "ksp_atol": 0.0,
             "ksp_max_it": 1000,
             "ksp_monitor": None,
             "ksp_converged_reason": None,
             "ksp_norm_type": "unpreconditioned",
             "pc_type": "python",
             "pc_python_type": "firedrake.PatchPC",
             "patch_pc_patch_save_operators": True,
             "patch_pc_patch_partition_of_unity": False,
             "patch_pc_patch_sub_mat_type": "seqaij",
             "patch_pc_patch_construct_type": "star",
             "patch_pc_patch_multiplicative": False,
             "patch_pc_patch_symmetrise_sweep": False,
             "patch_pc_patch_construct_dim": 0,
             "patch_sub_ksp_type": "preonly",
             "patch_sub_pc_type": "lu"}

w = Function(V1)

Prob = LinearVariationalProblem(a, F, w)

mg = True
if mg:
    Solver = LinearVariationalSolver(Prob, solver_parameters=mg_params)
    transfer = EmbeddedDGTransfer(V1.ufl_element(), use_fortin_interpolation=True)
    Solver.set_transfer_operators(dmhooks.transfer_operators(V1,
                                                             prolong=transfer.prolong,
                                                             inject=transfer.inject,
                                                             restrict=transfer.restrict))
else:
    Solver = LinearVariationalSolver(Prob, solver_parameters=patch_params)
Solver.solve()

f0 = File('hdiv-test.pvd')
f0.write(w)

