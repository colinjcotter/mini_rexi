from firedrake import *

ai = Constant(100)
ar = Constant(-1)

base = UnitSquareMesh(4,4)

nref = 5
mh = MeshHierarchy(base, nref)

meshes = []
for mesh in mh:
    x, y = SpatialCoordinate(mesh)
    V = VectorFunctionSpace(mesh, "CG", 1, dim=3)
    X = Function(V).interpolate(as_vector([x, y, x+y]))
    m2 = Mesh(X)
    m2.init_cell_orientations(as_vector([0,0,1]))
    meshes.append(m2)

embedded_mh = HierarchyBase(meshes, mh.coarse_to_fine_cells,
                            mh.fine_to_coarse_cells,
                            mh.refinements_per_level, mh.nested)
mesh = embedded_mh[-1]

x, y, z = SpatialCoordinate(mesh)

outward_normals = CellNormal(mesh)
perp = lambda u: cross(outward_normals, u)

V1 = FunctionSpace(mesh, "BDM", 2)
V2 = FunctionSpace(mesh, "DG", 1)
W = MixedFunctionSpace((V1, V1))

r = Function(V2)
r.interpolate(exp(x+y)*x*y)
File('r.pvd').write(r)

ur, ui = TrialFunctions(W)
vr, vi = TestFunctions(W)

f = Constant(0.1)  # Coriolis parameter
g = Constant(9.8)  # Gravitational constant
tau = Constant(10.)
H = Constant(1.0)

# (ar -ai)(hr) = tau*H*div(ur)
# (ai  ar)(hi)   tau*H*div(ui)
# (hr) = tau*H/(ar**2 + ai**2)(ar  ai)(div(ur))
# (hi)                        (-ai ar)(div(ui))

hr = tau*H/(ar**2 + ai**2)*(ar*div(ur) + ai*div(ui))
hi = tau*H/(ar**2 + ai**2)*(-ai*div(ur) + ar*div(ui))

a = (
    inner(ar*ur,vr) - inner(ai*ui, vr)
    - tau*inner(f*perp(ur),vr) + 
    tau*g*inner(hr,div(vr)) +
    inner(ar*ui,vi) + inner(ai*ur, vi)
    - tau*inner(f*perp(ui),vi) + 
    tau*g*inner(hi,div(vi))
)*dx

f1 = exp(x+y)*x*y
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
             "ksp_monitor_true_residual": None,
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
             "mg_levels_ksp_richardson_scale": 1/3,
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

w = Function(W)

Prob = LinearVariationalProblem(a, F, w)


mg = True
if mg:
    Solver = LinearVariationalSolver(Prob, solver_parameters=mg_params)

    transfer = EmbeddedDGTransfer(W.ufl_element(), use_fortin_interpolation=True)
    Solver.set_transfer_operators(dmhooks.transfer_operators(W,
                                                             prolong=transfer.prolong,
                                                             inject=transfer.inject,
                                                             restrict=transfer.restrict))
else:
    Solver = LinearVariationalSolver(Prob, solver_parameters=lu_params)
    
Solver.solve()

f0 = File('block.pvd')
ur, ui = w.split()
uout = Function(V1)
uout.assign(ur)
f0.write(uout)
uout.assign(ui)
f0.write(uout)

