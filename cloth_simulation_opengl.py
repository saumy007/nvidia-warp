import warp as wp
import numpy as np
import glfw
from OpenGL.GL import *

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

    v += gravity * dt

    for s in range(4):
        j = springs[i*4 + s]

        if j >= 0:
            d = pos[j] - p
            dist = wp.length(d)

            if dist > 0.0:
                f = stiffness * (dist - rest_length) * (d / dist)
                v += f * dt

    p += v * dt

    pos[i] = p
    vel[i] = v


# cloth grid
W, H = 30, 30
n = W * H

positions = []
for y in range(H):
    for x in range(W):
        positions.append(wp.vec3(x*0.1, 3.0, y*0.1))

pos = wp.array(positions, dtype=wp.vec3)
vel = wp.zeros(n, dtype=wp.vec3)

# neighbor springs
spr = np.full((n,4), -1, dtype=np.int32)

for y in range(H):
    for x in range(W):

        i = y*W + x

        if x>0: spr[i,0]=i-1
        if x<W-1: spr[i,1]=i+1
        if y>0: spr[i,2]=i-W
        if y<H-1: spr[i,3]=i+W

spr = wp.array(spr.flatten(), dtype=wp.int32)

dt = 0.016


# ---------- OPENGL ----------
glfw.init()
window = glfw.create_window(800,600,"GPU Cloth Simulation",None,None)
glfw.make_context_current(window)

glEnable(GL_POINT_SMOOTH)
glPointSize(4)


while not glfw.window_should_close(window):

    glfw.poll_events()

    wp.launch(
        simulate,
        dim=n,
        inputs=[pos, vel, spr, 0.1, 80.0, dt]
    )

    p = pos.numpy()

    glClear(GL_COLOR_BUFFER_BIT)

    glBegin(GL_POINTS)
    for v in p:
        glVertex3f(v[0], v[1]-2, v[2])
    glEnd()

    glfw.swap_buffers(window)

glfw.terminate()
