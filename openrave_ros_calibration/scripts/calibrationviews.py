from __future__ import with_statement # for python 2.5
__author__ = 'Rosen Diankov, Shohei Fujii'

import sys, os, time, threading
from openravepy import *
import openravepy
import numpy as np

from openravepy.misc import SpaceSamplerExtra
from contextlib import contextmanager
import json

import logging
log = logging.getLogger(__name__)

@contextmanager
def set_ode_collisionchecker(env):
    old = env.GetCollisionChecker()
    ode = RaveCreateCollisionChecker(env, 'ode')
    try:
        env.SetCollisionChecker(ode)
        yield
    finally:
        env.SetCollisionChecker(old)

class CalibrationViews(object):
    def __init__(self,robot,sensorname=None,sensorrobot=None,target=None):
        self.env = robot.GetEnv()
        self.robot = robot
        if target is None:
            raise ValueError('target is empty')
        self.vmodel = databases.visibilitymodel.VisibilityModel(robot=robot,sensorrobot=sensorrobot,targetlink=target,sensorname=sensorname)
        self.vmodel.load()
        self.Tpatternrobot = None
        if self.vmodel.robot != self.vmodel.sensorrobot and target is not None:
            log.info('Assuming target \'%s\' is attached to %s'%(target.GetName(),self.vmodel.manip))
            self.Tpatternrobot = np.dot(np.linalg.inv(self.vmodel.targetlink.GetTransform()),self.vmodel.manip.GetEndEffectorTransform())

    def computevisibilityposes(self,dists=np.arange(0.05,1.5,0.2),orientationdensity=1,num=np.inf):
        """Computes robot poses using visibility information from the target.

        Sample the transformations of the camera. the camera x and y axes should always be aligned with the 
        xy axes of the calibration pattern.
        """
        with set_ode_collisionchecker(self.env):
            if not self.vmodel.has():
                # nothing is loaded
                self.vmodel.visibilitytransforms = self.vmodel.visualprob.ProcessVisibilityExtents(numrolls=8,sphere=[orientationdensity]+dists.tolist())
                self.vmodel.preshapes = np.array([self.robot.GetDOFValues(self.vmodel.manip.GetGripperIndices())])
                self.vmodel.preprocess()
            # if self.Tpatternrobot is not None:
            #     self.vmodel.targetlink.SetTransform(np.dot(self.vmodel.manip.GetEndEffectorTransform(),np.linalg.inv(self.Tpatternrobot)))
            #with RobotStateSaver(self.robot,KinBody.SaveParameters.GrabbedBodies):
            if True:
                ab = self.vmodel.targetlink.ComputeAABBFromTransform(np.eye(4))
                centers = np.dot(np.array(((0,0,0),(0.5,0.5,0),(-0.5,0.5,0),(0.5,-0.5,0),(-0.5,-0.5,0))), np.diag(ab.extents()))
                if self.Tpatternrobot is not None:
                    # self.robot.Grab(self.vmodel.targetlink,self.vmodel.manip.GetEndEffector())
                    Tbase = self.vmodel.attachedsensor.GetTransform()
                    visibilitytransforms = invertPoses(self.vmodel.visibilitytransforms)
                else:
                    Tbase = self.vmodel.targetlink.GetTransform()
                    visibilitytransforms = self.vmodel.visibilitytransforms
                posebase = poseFromMatrix(Tbase)
                poses = []
                configs = []
                for relativepose in visibilitytransforms:
                    for center in centers:
                        if self.Tpatternrobot is not None:
                            pose = np.array(posebase)
                            pose[4:7] += quatRotate(pose[0:4],center)
                            pose = poseMult(pose,relativepose)
                        else:
                            pose = poseMult(posebase,relativepose)
                            pose[4:7] += quatRotate(pose[0:4],center)
                        try:
                            q = self.vmodel.visualprob.ComputeVisibleConfiguration(pose=pose)
                            ret = json.loads(q)
                            if 'error' in ret:
                                log.debug('invalid pose: %s'%ret['error']['type'])
                                continue
                            poses.append(pose)
                            configs.append(ret['solution'])
                            if len(poses) > num:
                                return np.array(poses), np.array(configs)
                        except planning_error:
                            pass
                return np.array(poses), np.array(configs)

    def viewVisibleConfigurations(self, poses, configs):
        graphs = [self.env.drawlinelist(np.array([pose[4:7],pose[4:7]+0.03*rotationMatrixFromQuat(pose[0:4])[0:3,2]]),1) for pose in poses]
        try:
            with self.robot:
                for i,config in enumerate(configs):
                    self.robot.SetDOFValues(config,self.vmodel.manip.GetArmIndices())
                    self.env.UpdatePublishedBodies()
                    raw_input('%d: press any key'%i)
        finally:
            graphs = None

if __name__ == '__main__':
    import sys
    env = Environment()
    env.Load(sys.argv[1])
    env.SetViewer('qtosg')

    robot = env.GetRobot('Denso')
    sensorrobot = env.GetRobot('azure_kinect')
    sensorname = 'rgb_camera'
    target = robot.GetLink('pattern')
    self = CalibrationViews(robot=robot, sensorname=sensorname, sensorrobot=sensorrobot, target=target)
    poses, configs = self.computevisibilityposes(dists=np.arange(0.03,1.5,0.2), orientationdensity=5, num=1000)
    # self.viewVisibleConfigurations(poses, configs)
    sensor = self.vmodel.attachedsensor.GetSensor()
    camerageom = sensor.GetSensorGeometry(Sensor.Type.Camera)
    from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())


