import mujoco
import glfw
import numpy as np
# np.set_printoptions(precision=4)
import math
# import matplotlib.pyplot as plt

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

model = mujoco.MjModel.from_xml_path('model/InvertedPendulum.xml')
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
camera.distance = 2
camera.azimuth = -45
camera.elevation = -45

mujoco.mjv_updateScene(
    model, data, mujoco.MjvOption(), mujoco.MjvPerturb(),
    camera, mujoco.mjtCatBit.mjCAT_ALL, scene)

set_point = np.array([45])
data.qpos = set_point

while(not glfw.window_should_close(window)):
    # mujoco.mj_step1(model, data)
    # # update data.ctrl to move robot
    # mujoco.mj_step2(model, data)
    mujoco.mj_step(model, data)

    mujoco.mjv_updateScene(
        model, data, mujoco.MjvOption(), None,
        camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()