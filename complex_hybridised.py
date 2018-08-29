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
alpha = Constant(1+1j)

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

u0 = Function(M)

#solve(a==L, u0,
#      solver_parameters={'ksp_type':'gmres',
#                         'pc_type':'gamg'})

#File('u0.pvd').write(u0)
