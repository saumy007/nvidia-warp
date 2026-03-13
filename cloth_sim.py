import warp as wp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

wp.init()

gravity = wp.vec3(0.0, -9.8, 0.0)

@wp.kernel
def simulate(pos: wp.array(dtype=wp.vec3),
             vel: wp.array(dtype=wp.vec3),
             springs: wp.array(dtype=wp.int32),
             rest_length: float,
             stiffness: float,
             dt: float):

    i = wp.tid()

    p = pos[i]
    v = vel[i]

    # gravity
    v += gravity * dt

    # spring forces
    for s in range(4):
        j = springs[i*4 + s]

        if j >= 0:
            dir = pos[j] - p
            dist = wp.length(dir)

            if dist > 0.0:
                force = stiffness * (dist - rest_length) * (dir / dist)
                v += force * dt

    p += v * dt

    pos[i] = p
    vel[i] = v


# cloth grid
width = 15
height = 15
n = width * height

positions = []
for y in range(height):
    for x in range(width):
        positions.append(wp.vec3(x*0.2, 5.0, y*0.2))

pos = wp.array(positions, dtype=wp.vec3)
vel = wp.zeros(n, dtype=wp.vec3)

# springs
springs = np.full((n,4), -1, dtype=np.int32)

for y in range(height):
    for x in range(width):

        i = y*width + x

        if x > 0:
            springs[i,0] = i-1
        if x < width-1:
            springs[i,1] = i+1
        if y > 0:
            springs[i,2] = i-width
        if y < height-1:
            springs[i,3] = i+width

springs = wp.array(springs.flatten(), dtype=wp.int32)

dt = 0.016


# ---------- UI SETUP ----------
fig, ax = plt.subplots()
scatter = ax.scatter([], [])

ax.set_xlim(-1, width*0.2 + 1)
ax.set_ylim(0, 6)


def update(frame):

    wp.launch(
        simulate,
        dim=n,
        inputs=[pos, vel, springs, 0.2, 50.0, dt]
    )

    data = pos.numpy()

    x = data[:,0]
    y = data[:,1]

    scatter.set_offsets(np.c_[x,y])

    return scatter,


ani = FuncAnimation(fig, update, interval=16)

plt.title("Real-Time Cloth Simulation (GPU Warp)")
plt.show()
