from firedrake import *

n = 200
mesh = UnitSquareMesh(n,n)

V = FunctionSpace(mesh,"CG",1)

v = TestFunction(V)
u = TrialFunction(V)

b = Constant(1+1.5j)

a = ((1+1j)*inner(u,v) + inner(b*grad(u), grad(v)))*dx

x, y = SpatialCoordinate(mesh)

from numpy import pi
f1 = sin(2*pi*x)*sin(2*pi*y)*exp(-10*(x**2 + y**2))
L = inner(f1,v)*dx

assemble(L)

u0 = Function(V)

solve(a==L, u0,
      solver_parameters={'ksp_type':'gmres',
                         'ksp_monitor':True,
                         'pc_type':'gamg'})

File('u0.pvd').write(u0)
