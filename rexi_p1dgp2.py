from firedrake import *

n = 20
mesh = UnitSquareMesh(n,n)

#we are using P1dg-P2 for the time being, it's not what we want but is what
#currently works with complex-valued fields

V = VectorFunctionSpace(mesh,"DG",1)
Q = FunctionSpace(mesh, "CG", 2)

perp = lambda u: as_vector([-u[1], u[0]])
dt = Constant(1.0)
g = Constant(1.0)
f = Constant(1.0)
H = Constant(1.0)
alpha = Constant(1)
beta = Constant(1)

# L(u) = -f*perp(u) - g grad h
#  (h) = - H*div(u)

# perp(u) = (0 -1)
#           (1, 0)u := R*u

# (a_i*I + dt*L)(v_i,q_i)^T = (u_0,h_0)^T
# then assemble (u_1,h_1) = sum_i b_i (v_i, q_i)

# elimination of v_i,
# (a_i*I + f*dt*R)v_i = u_0 - dt*g*grad(h)
# v_i = -dt*(a_i*I +f*dt*R)^{-1}g*grad(h) + (a_i*I + f*dt*R)^{-1}u_0.

# (a_i*I + dt*f*R)^{-1} = (a_i -dt*f)^{-1} = 1/(a_i**2 + dt**2*f**2)*(a_i   dt*f)
#                         (dt*f  a_i)                                (-dt*f  a_i)

# a_i*q - dt**2*g*H*div((a_i*I +dt*f*R)^{-1}g*grad(h)) = h_0 - dt*H*div(a_i*I + dt*f*R)^{-1}u_0

aiIpRinv = TensorConstant([[alpha,dt*f],[-dt*f,alpha]])/(dt**2*f**2+alpha**2)

x, y = SpatialCoordinate(mesh)
u0 = Function(V)
h0 = Function(Q).interpolate(cos(2*pi*x))

m1 = Function(V)
h1 = Function(Q)

q = TrialFunction(Q)
p = TestFunction(Q)

ah = a_i*inner(q,p)*dx + dt**2*g*H*inner(aiIpRinv*grad(q), grad(p))*dx
Lh = inner(h0,p)*dx + dt*H*inner(aiIpRinv*grad(q), u0)*dx

hparams = {'ksp_type':'gmres',
           'pc_type':'bjacobi',
           'sub_pc_type':'ilu'}

hProblem = LinearVariationalProblem(ah, Lh, h1)
hSolver = LinearVariationalSolver(hProblem, solver_parameters=hparams)

v = TrialFunction(V)
w = TestFunction(V)

uparams = {'ksp_type':'gmres',
           'pc_type':'bjacobi',
           'sub_pc_type':'ilu'}

au = inner(w,a_i*v + dt*f*perp(v))*dx
Lu = inner(w,u0)*dx - dt*g*inner(w,grad(h1))*dx

uProblem = LinearVariationalProblem(au, Lu, u1)
uSolver = LinearVariationalSolver(uProblem, solver_parameters=uparams)

nsteps = 10

for step in range(nsteps):
    hSolver.solve()
    uSolver.solve()
    u1 *= beta
    h1 *= beta
    u0.assign(u1)
    h0.assign(h1)
