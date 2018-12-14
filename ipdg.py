from firedrake import *

ai = Constant(1.0 + 1j)

f = Constant(1.0e-4)  # Coriolis parameter
g = Constant(9.8)  # Gravitational constant
H = Constant(1000.)  # Mean depth
tau = Constant(10.)

nx = 32
mesh = PeriodicUnitSquareMesh(nx, nx)
V1 = FunctionSpace(mesh, "BDM", 2)
V2 = FunctionSpace(mesh, "DG", 1)
W = MixedFunctionSpace((V1, V2))

u, h = TrialFunctions(W)
v, phi = TestFunctions(W)

a = (
    inner(ai*u,v) + tau*inner(f*perp(u),v) + 
    g*tau*inner(h,div(v))
    +inner(ai*h,phi) - H*tau*inner(div(u),phi)
)*dx

class Helm(ExplicitSchurPC):

    def form(self, pc, test, trial):
        h = trial
        phi = test

        n = FacetNormal(mesh)
        alpha = Constant(5)
        CellVol = CellVolume(mesh)
        FaceArea = FacetArea(mesh)
        dx_avg = avg(CellVol)/FaceArea

        Y = ai*tau**2*g*H/(ai**2 + (tau*f)**2)
        
        Pform =(
            inner(ai*h,phi)*dx
            + Y*(
                inner(grad(h), grad(phi))*dx
                - inner(avg(grad(h)), jump(phi, n))*dS
                - inner(jump(h, n), avg(grad(phi)))*dS
                + alpha/dx_avg*inner(jump(h, n), jump(phi, n))*dS
            )
        )
        bcs = None
        return (Pform, bcs)

x, y = SpatialCoordinate(mesh)
import math
f1 = sin(2*math.pi*x)*sin(2*math.pi*y)*exp(-10*(x**2 + y**2))

F = inner(f1,phi)*dx

params = {
    "ksp_type": "fgmres",
    "ksp_monitor": True,
    "ksp_rtol": 1e-8,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "lu",
    "fieldsplit_1_ksp_type": "gmres",
    "fieldsplit_1_pc_type": "python",
    "fieldsplit_1_pc_python_type": "__main__.Helm",
    "fieldsplit_1_schur_pc_type":"ilu",
    "fieldsplit_1_ksp_atol":1.0e-6,
    "fieldsplit_1_ksp_monitor": True,
    "fieldsplit_0_ksp_converged_reason": True,
    "ksp_converged_reason": True
}

w = Function(W)

Prob = LinearVariationalProblem(a, F, w)
Solver = LinearVariationalSolver(Prob, solver_parameters=params)

Solver.solve()
