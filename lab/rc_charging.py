
import math

# RC Circuit Parameters
Vs = 10.0
R = 1000.0
C = 1000e-6

V0 = 0.0
h = 0.2
t_end = 5.0

# Time constant
tau = R * C

# Differential equation
def f(t, vc):
    return (Vs - vc) / tau


# Number of time steps
steps = int(round(t_end / h))

# Arrays for solutions
times = [i * h for i in range(steps + 1)]

euler = [0.0] * (steps + 1)
rk4 = [0.0] * (steps + 1)

euler[0] = V0
rk4[0] = V0


# Euler Method
for n in range(steps):

    t = times[n]
    vc = euler[n]

    euler[n + 1] = vc + h * f(t, vc)


# Runge-Kutta 4 Method
for n in range(steps):

    t = times[n]
    vc = rk4[n]

    k1 = f(t, vc)
    k2 = f(t + h/2, vc + h*k1/2)
    k3 = f(t + h/2, vc + h*k2/2)
    k4 = f(t + h, vc + h*k3)

    rk4[n + 1] = vc + (h/6) * (
        k1 + 2*k2 + 2*k3 + k4
    )


# Exact Analytical Solution
exact = [
    Vs * (1 - math.exp(-t / tau))
    for t in times
]


# Display Results
print("RC CHARGING CIRCUIT")
print("------------------------------")
print("Source voltage:", Vs, "V")
print("Resistance:", R, "Ohm")
print("Capacitance:", C, "F")
print("Time constant:", tau, "s")
print("Time step:", h, "s")
print("Total steps:", steps)

print("\nNumerical Results")
print("----------------------------------------------")
print("Time(s)   Euler(V)   RK4(V)   Exact(V)")

for i in range(steps + 1):

    print(
        f"{times[i]:5.1f}    "
        f"{euler[i]:8.6f}  "
        f"{rk4[i]:8.6f}  "
        f"{exact[i]:8.6f}"
    )


# Independent Error Check
euler_errors = [
    abs(euler[i] - exact[i])
    for i in range(steps + 1)
]

rk4_errors = [
    abs(rk4[i] - exact[i])
    for i in range(steps + 1)
]

print("\nIndependent Check")
print("------------------------------")

print(
    "Maximum Euler error:",
    f"{max(euler_errors):.8f} V"
)

print(
    "Maximum RK4 error:",
    f"{max(rk4_errors):.8f} V"
)

print("\nFinal Values at 5 seconds")
print("------------------------------")
print(f"Euler: {euler[-1]:.6f} V")
print(f"RK4: {rk4[-1]:.6f} V")
print(f"Exact: {exact[-1]:.6f} V")
