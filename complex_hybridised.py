from firedrake import *

n = 20
mesh = UnitSquareMesh(n,n)

V = FunctionSpace(mesh,"BDM",1)
Q = FunctionSpace(mesh, "DG", 0)

M = MixedFunctionSpace((V,Q))

w, q = TestFunctions(M)
u, h = TrialFunctions(M)

perp = lambda u: as_vector([-u[1], u[0]])
dt = Constant(1.0)
g = Constant(1.0)
f = Constant(1.0)
H = Constant(1.0)
#alpha = Constant(1+1j)
alpha = Constant(1)

a = (
    inner(alpha*u,w)
    - dt*f*inner(perp(u), w)
    + dt*inner(h,div(w))
    + inner(alpha*h,q)
    - dt*H*inner(div(u),q)
)*dx

assemble(a)

x, y = SpatialCoordinate(mesh)
f = Function(Q).interpolate(cos(2*pi*x))
L = inner(f,q)*dx

assemble(L)

m0 = Function(M)

lu_parameters = {'mat_type':'aij',
                 'ksp_type':'preonly',
                 'pc_type':'lu',
                 'pc_factor_mat_solver_type': 'mumps'}

hyb_parameters = {'mat_type': 'matfree',
                  'ksp_type': 'preonly',
                  'pc_type': 'python',
                'pc_python_type': 'firedrake.HybridizationPC',
                  'hybridization': {'ksp_type': 'preonly',
                                    'pc_type': 'lu',
                                    'pc_factor_mat_solver_type': 'mumps'}}

solve(a==L, m0, solver_parameters=lu_parameters)

u0 = Function(V)
q0 = Function(Q)

u1, q1 = m0.split()
u1.assign(u0)
q1.assign(q0)
File('lu.pvd').write(u1,q1)

solve(a==L, m0, solver_parameters=hyb_parameters)
u0 = Function(V)
q0 = Function(Q)

u1, q1 = m0.split()
u1.assign(u0)
q1.assign(q0)
File('hyb.pvd').write(u1,q1)
