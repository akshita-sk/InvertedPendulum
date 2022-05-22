import mujoco
import glfw
import numpy as np
# np.set_printoptions(precision=4)
import matplotlib.pyplot as plt

def init_window(max_width, max_height):
    glfw.init()
    window = glfw.create_window(width=max_width, height=max_height,
                                       title='Demo', monitor=None,
                                       share=None)
    glfw.make_context_current(window)
    return window

window = init_window(1080, 720)
width, height = glfw.get_framebuffer_size(window)
viewport = mujoco.MjrRect(0, 0, width, height)

model = mujoco.MjModel.from_xml_path('/home/akshita/Documents/InvertedPendulum/model/InvertedPendulum.xml')
# model = mujoco.MjModel.from_xml_path('/home/akshita/Documents/InvertedPendulum/model/doublependulum.xml')
# model.nuserdata = 25
data = mujoco.MjData(model)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

scene = mujoco.MjvScene(model, 6000)
camera = mujoco.MjvCamera()
# base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base')
camera.trackbodyid = 1
# camera.distance = 1
# camera.azimuth = 0
# camera.elevation = -10
camera.distance = 3
camera.azimuth = 90
camera.elevation = -10

mujoco.mjv_updateScene(
    model, data, mujoco.MjvOption(), mujoco.MjvPerturb(),
    camera, mujoco.mjtCatBit.mjCAT_ALL, scene)

Kp = 10000
Kd = (np.sqrt(Kp)/2) + 193
data.qpos = np.array([np.deg2rad(45)])

states = np.zeros((1,model.nq+model.nv+1))
states[0,:] = np.hstack((data.time, data.qpos, data.qvel))
cstate = np.zeros((1,model.nq+model.nv+1))

while(not glfw.window_should_close(window)):
    mujoco.mj_step1(model, data)
    
    # update data.ctrl to move robot
    data.ctrl = Kd*(np.zeros(model.nv) - data.qvel) + Kp*(np.zeros(model.nq) - data.qpos)
    
    mujoco.mj_step2(model, data)
    # mujoco.mj_step(model, data)
    
    cstate[0,:] = np.hstack((data.time, data.qpos, data.qvel))
    states = np.append(states, cstate, axis=0)
    
    mujoco.mjv_updateScene(
        model, data, mujoco.MjvOption(), None,
        camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()

plt.plot(states[:,0], states[:,1], color='b', label='Position')
plt.plot(states[:,0], states[:,2], color='r', label='Velocity')
plt.legend()
plt.title('Evolution of States')
plt.show()