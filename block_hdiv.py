from firedrake import *

R = 6371220.
H = Constant(5960.)

base = IcosahedralSphereMesh(radius=R, refinement_level=0)
nref = 5
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
W = MixedFunctionSpace((V1, V1))

ur, ui = TrialFunctions(W)
vr, vi = TestFunctions(W)

Omega = Constant(7.292e-5)  # rotation rate
R = Constant(R)
f = Constant(2.0)*Omega*z/R  # Coriolis parameter
g = Constant(9.8)  # Gravitational constant
hour = 60*60
hours = 12
tau = Constant(hour*hours)

aval = 0.1 + 30.j
ai = Constant(imag(aval))
ar = Constant(real(aval))

bval = aval**2
#set this to 0.5 for optimal operator, 1 for optimal multigrid
bpow = 0.5
bmod = real(bval) + (imag(bval) + (-real(bval))**bpow)*1j
aPval = bmod**0.5

aiP = Constant(imag(aPval))
arP = Constant(real(aPval))


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

hrP = tau*H/(arP**2 + aiP**2)*(arP*div(ur) + aiP*div(ui))
hiP = tau*H/(arP**2 + aiP**2)*(-aiP*div(ur) + arP*div(ui))

aP = (
    inner(arP*ur,vr) - inner(aiP*ui, vr)
    - tau*inner(f*perp(ur),vr) + 
    tau*g*inner(hr,div(vr)) +
    inner(arP*ui,vi) + inner(aiP*ur, vi)
    - tau*inner(f*perp(ui),vi) + 
    tau*g*inner(hi,div(vi))
)*dx


f1 = exp((x+y+z)/R)*x*y*z/R**3
F = inner(div(vr),f1)*dx

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

class Shifted(AuxiliaryOperatorPC):

    def form(self, pc, test, trial):

        ur, ui = split(trial)
        vr, vi = split(test)
        
        hrP = tau*H/(arP**2 + aiP**2)*(arP*div(ur) + aiP*div(ui))
        hiP = tau*H/(arP**2 + aiP**2)*(-aiP*div(ur) + arP*div(ui))
        
        aP = (
            inner(arP*ur,vr) - inner(aiP*ui, vr)
            - tau*inner(f*perp(ur),vr) + 
            tau*g*inner(hr,div(vr)) +
            inner(arP*ui,vi) + inner(aiP*ur, vi)
            - tau*inner(f*perp(ui),vi) + 
            tau*g*inner(hi,div(vi))
        )*dx

        bcs = None
        return (aP, bcs)

ap_mg_params = {"mat_type": "matfree",
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
                "mg_levels_ksp_richardson_scale": 1/3,
                "mg_levels_pc_type": "python",
                "mg_levels_pc_python_type": "__main__.Shifted",
                "mg_levels_aux_pc_type": "firedrake.PatchPC",
                "mg_levels_aux_patch_pc_patch_save_operators": True,
                "mg_levels_aux_patch_pc_patch_partition_of_unity": False,
                "mg_levels_aux_patch_pc_patch_sub_mat_type": "seqaij",
                "mg_levels_aux_patch_pc_patch_construct_type": "star",
                "mg_levels_aux_patch_pc_patch_multiplicative": False,
                "mg_levels_aux_patch_pc_patch_symmetrise_sweep": False,
                "mg_levels_aux_patch_pc_patch_construct_dim": 0,
                "mg_levels_aux_patch_sub_ksp_type": "preonly",
                "mg_levels_aux_patch_sub_pc_type": "lu"}

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
             "patch_pc_patch_construct_type": "vanka",
             "patch_pc_patch_multiplicative": False,
             "patch_pc_patch_symmetrise_sweep": False,
             "patch_pc_patch_construct_dim": 0,
             "patch_sub_ksp_type": "preonly",
             "patch_sub_pc_type": "lu"}

w = Function(W)

mg = True
if mg:
    Prob = LinearVariationalProblem(a, F, w)
    Solver = LinearVariationalSolver(Prob,
                                     solver_parameters=mg_params)
    transfer = EmbeddedDGTransfer(W.ufl_element(),
                                  use_fortin_interpolation=True)
    Solver.set_transfer_operators(dmhooks.transfer_operators(W,
                                                             prolong=transfer.prolong,
                                                             inject=transfer.inject,
                                                             restrict=transfer.restrict))
else:
    Prob = LinearVariationalProblem(a, F, w, aP=aP)
    Solver = LinearVariationalSolver(Prob, solver_parameters=lu_params)
Solver.solve()

f0 = File('block.pvd')
ur, ui = w.split()
uout = Function(V1)
uout.assign(ur)
f0.write(uout)
uout.assign(ui)
f0.write(uout)

