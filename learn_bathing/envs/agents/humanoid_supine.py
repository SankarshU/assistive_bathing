"""
Portions of this file are derived from Bullet Physics (bullet3) code:
  https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/deep_mimic/mocap/humanoid.py

Original Code License (zlib):
---------------------------------------------------------------------------
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2014 Erwin Coumans

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the
use of this software. Permission is granted to anyone to use this software
for any purpose, including commercial applications, and to alter it and
redistribute it freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
---------------------------------------------------------------------------

Modifications made by Yubin Koh (koh22@purdue.edu)
"""


import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


class Humanoid(object):

  def __init__(self, pybullet_client, baseShift, ornShift=[0,0,0,1]):
    """Constructs a humanoid and reset it to the initial states.
    Args:
      pybullet_client: The instance of BulletClient to manage different
        simulations.
    """
    self._baseShift = baseShift
    self._ornShift = ornShift
    self._pybullet_client = pybullet_client

    self._humanoid = self._pybullet_client.loadURDF("learn_bathing/envs/urdf/humanoid_with_rev.urdf", [0, 0.9, 0],
                                                    globalScaling=0.2,
                                                    useFixedBase=True)

    self._pybullet_client.resetBasePositionAndOrientation(self._humanoid, self._baseShift, self._ornShift)
    self._pybullet_client.changeDynamics(self._humanoid, -1, linearDamping=0, angularDamping=0)
    
    # change colors of the human model limbs
    humanoid_color = [255/255, 160/255, 45/255, 1]
    for j in range(self._pybullet_client.getNumJoints(self._humanoid)):  
      ji = self._pybullet_client.getJointInfo(self._humanoid, j)
      self._pybullet_client.changeDynamics(self._humanoid, j, linearDamping=0, angularDamping=0)
      self._pybullet_client.changeVisualShape(self._humanoid, j, rgbaColor=humanoid_color)

    self._initial_state = self._pybullet_client.saveState()
    self._allowed_body_parts = [11, 14]
    
    self._contact_point = [0, 0, 0, 0]
