from firedrake import *

from rexi_coefficients import RexiCoefficients

h = 0.2
M = 32
alphas, betas = RexiCoefficients(h, M, 0, reduce_to_half=True)

n = 20
ensemble = Ensemble(COMM_WORLD, 2)

mesh = UnitSquareMesh(20, 20, comm=ensemble.comm)

#we are using P1dg-P2 for the time being, it's not what we want but is what
#currently works with complex-valued fields

V = VectorFunctionSpace(mesh,"DG",1)
Q = FunctionSpace(mesh, "CG", 2)

perp = lambda u: as_vector([-u[1], u[0]])
dt = Constant(0.01)
g = Constant(1.0)
f = Constant(1.0)
H = Constant(1.0)
alpha = Constant(alphas[ensemble.ensemble_comm.rank])
beta = Constant(betas[ensemble.ensemble_comm.rank])

# L(u) = -f*perp(u) - g grad h
#  (h) = - H*div(u)

# perp(u) = (0 -1)
#           (1, 0)u := R*u

# (a_i*I + dt*L)(v_i,q_i)^T = (u_0,h_0)^T
# then assemble (u_1,h_1) = sum_i b_i (v_i, q_i)

# elimination of v_i,
# (a_i*I - f*dt*R)v_i = u_0 + dt*g*grad(h)
# v_i = dt*(a_i*I  - f*dt*R)^{-1}g*grad(h) + (a_i*I - f*dt*R)^{-1}u_0.

# (a_i*I - dt*f*R)^{-1} = (a_i   dt*f)^{-1} = 1/(a_i**2 + dt**2*f**2)*(a_i   -dt*f)
#                         (-dt*f  a_i)                                (dt*f    a_i)

# a_i*q - dt**2*g*H*div((a_i*I - dt*f*R)^{-1}g*grad(h)) = h_0 + dt*H*div(a_i*I - dt*f*R)^{-1}u_0

aiIpRinv = as_matrix([[alpha,-dt*f],[dt*f,alpha]])/(dt**2*f**2+alpha**2)


x, y = SpatialCoordinate(mesh)
u0 = Function(V)
h0 = Function(Q).interpolate(cos(2*pi*x))

u1 = Function(V)
h1 = Function(Q)

u1T = Function(V)
h1T = Function(Q)

q = TrialFunction(Q)
p = TestFunction(Q)

ah = alpha*inner(q,p)*dx + dt**2*g*H*inner(aiIpRinv*grad(q), grad(p))*dx
Lh = inner(h0,p)*dx + dt*H*inner(u0, aiIpRinv*grad(p))*dx

hparams = {'ksp_type':'gmres',
           'ksp_monitor':True,
           'pc_type':'bjacobi',
           'sub_pc_type':'ilu'}

hProblem = LinearVariationalProblem(ah, Lh, h1)
hSolver = LinearVariationalSolver(hProblem, solver_parameters=hparams)

v = TrialFunction(V)
w = TestFunction(V)

uparams = {'ksp_type':'gmres',
           'ksp_monitor':True,
           'pc_type':'bjacobi',
           'sub_pc_type':'ilu'}

au = inner(alpha*v - dt*f*perp(v), w)*dx
Lu = inner(u0, w)*dx - dt*g*inner(grad(h1), w)*dx

uProblem = LinearVariationalProblem(au, Lu, u1)
uSolver = LinearVariationalSolver(uProblem, solver_parameters=uparams)

nsteps = 10

for step in range(nsteps):
    hSolver.solve()
    uSolver.solve()
    u1 *= beta
    h1 *= beta
    ensemble.allreduce(u1, u1T)
    ensemble.allreduce(h1, h1T)
    u0.assign(u1T)
    h0.assign(h1T)
