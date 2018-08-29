from firedrake import *

n = 20
mesh = UnitSquareMesh(n,n)

V = FunctionSpace(mesh,"CG",1)

v = TestFunction(V)
u = TrialFunction(V)

a = ((1+1j)*inner(u,v) + inner(grad(u), grad(v)))*dx
assemble(a)

x, y = SpatialCoordinate(mesh)

f = Function(V).interpolate(cos(2*pi*x))
L = inner(f,v)*dx

assemble(L)

u0 = Function(V)

solve(a==L, u0,
      solver_parameters={'ksp_type':'gmres',
                         'pc_type':'gamg'})

File('u0.pvd').write(u0)
