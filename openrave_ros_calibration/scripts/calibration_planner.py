import rospy
from openravepy import *
import numpy as np

class OpenRAVECalibrationPlanner(object):
    def __init__(self, ):
        envfile = rospy.param('collada_file')
        robotname = rospy.param('robot')
        manipname = rospy.param('manip', None)
        targetname = rospy.param('target')
        sensorobotname = rospy.param('sensorrobot')
        self.sensorname = rospy.param('sensorname')
        viewername = rospy.param('viewer', 'qtosg')
        controllerargs = rospy.param('controllerargs', None)

        self.env = Environment()
        self.env.SetViewer(viewername)
        self.env.Load(envfile)
        self.robot = self.env.GetRobot(robotname)
        if manipname is None:
            manipname = self.robot.GetActiveManipulator().GetName()
        self.manip = self.robot.GetManipulator(manipname)
        self.sensorrobot = self.env.GetRobot(sensorrobotname)
        self.sensorrobot.GetAttachedSensor(self.sensorname)
        self.target = robot.GetLink(targetname)

        if controllerargs is not None:
            controller = RaveCreateController(self.env, controllerargs)
            self.robot.SetController(controller, list(self.robot.GetActiveDOFIndices()), 0)

        self.basemanip = interfaces.BaseManipulation(manip=self.manip)
        lmodel = databases.linkstatistics.LinkStatisticsModel(self.robot)
        if not lmodel.load():
            lmodel.autogenerate()
        lmodel.setRobotWeights()
        lmodel.setRobotResolutions()
        # ikmodel = databases.inversekinematics.InverseKinematicsModel(robot=self.robot, manip=self.manip, iktype=IkParameterizationType.Transform6D)


    def computePoses(self, dists=np.arange(0.03,1.5,0.2), orientationdensity=5, num=np.inf):
        self.calibrationviews = CalibrationViews(robot=self.robot, sensorname=self.sensorname, sensorrobot=self.sensorrobot, target=self.target)
        poses, configs = self.computevisibilityposes(dists=dists, orientationdensity=orientationdensity, num=num)
        return poses, configs


if __name__ == '__main__':
    rospy.init_node('openrave_calibration_planner')
    # self.viewVisibleConfigurations(poses, configs)
    from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())

