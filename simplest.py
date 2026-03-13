import warp as wp

wp.init()

# gravity vector
gravity = wp.vec3(0.0, -9.8, 0.0)

# physics kernel
@wp.kernel
def simulate(pos: wp.array(dtype=wp.vec3),
             vel: wp.array(dtype=wp.vec3),
             dt: float):

    i = wp.tid()

    # update velocity
    vel[i] = vel[i] + gravity * dt

    # update position
    pos[i] = pos[i] + vel[i] * dt


# number of particles
n = 5

# initial positions
positions = wp.array([
    wp.vec3(0.0, 10.0, 0.0),
    wp.vec3(1.0, 12.0, 0.0),
    wp.vec3(2.0, 14.0, 0.0),
    wp.vec3(3.0, 16.0, 0.0),
    wp.vec3(4.0, 18.0, 0.0)
], dtype=wp.vec3)

# initial velocities
velocities = wp.zeros(n, dtype=wp.vec3)

dt = 0.016  # 60 FPS timestep

# run simulation for 10 steps
for step in range(10):

    wp.launch(
        kernel=simulate,
        dim=n,
        inputs=[positions, velocities, dt]
    )

    print("step", step)
    print(positions.numpy())
