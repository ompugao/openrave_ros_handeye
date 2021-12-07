import rospy
from openravepy import *
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
import time
import threading
import cv2
import handeye

from cv_bridge import CvBridge

class OpenRAVECalibrationPlanner(object):
    def __init__(self, env, robotname, manipname,
            targetname, sensorobotname, sensorname, viewername, controllerargs):
        self.env = env
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

    def moveTo(self, config):
        self.basemanip.MoveActiveJoints(goal=config)
        while not self.robot.GetController().IsDone():
            time.sleep(1.0)


class CalibrationTaskController(object  ):

    def __init__(self, ):
        envfile = rospy.get_param('~collada_file')
        robotname = rospy.get_param('~robot')
        manipname = rospy.get_param('~manip', None)
        targetname = rospy.get_param('~target')
        sensorobotname = rospy.get_param('~sensorrobot')
        self.sensorname = rospy.get_param('~sensorname')
        viewername = rospy.get_param('~viewer', 'qtosg')
        controllerargs = rospy.get_param('~controllerargs', None)

        env = Environment()
        env.Load(envfile)
        self.calibplanner = OpenRAVECalibrationPlanner(self.env, robotname, manipname,
            targetname, sensorobotname, sensorname, viewername, controllerargs)

        self.gray_image = None
        self.cam_info = None
        self.image_lock = threading.Lock()
        self.caminfo_lock = threading.Lock()

        self.cv_bridge = CvBridge()
        self.image_sub = rospy.Subscriber("~input/image_rect", Image, self._image_callback)
        self.camera_info_sub = rospy.Subscriber("~input/camera_info", CameraInfo, self._camera_info_callback)
        self.image_pub = rospy.Publisher("~output/debug_image_calib", Image)


    def _image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            return

        (rows,cols,channels) = cv_image.shape
        with self.image_lock:
            self.gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _camera_info_callback(self, msg):
        with self.caminfo_lock:
            self.cam_info = msg

    def do_calibration(self, observations):
        setup = handeye.calibrator.Setup.Fixed
        calibrator = handeye.calibrator.HandEyeCalibrator(setup)
        for i, obs in enumerate(observations):
            Q = obs['Tmanipinbase']
            P = obs['Tpatternincamera']
            calibrator.add_sample(Q, P)
        Xhat = calibrator.solve(method=handeye.solver.ParkBryan1994)

    def gather_observations(self, dists=np.arange(0.03,1.5,0.2), orientationdensity=5, num=np.inf):
        poses, configs = self.calibplanner.computePoses(self, dists=dists, orientationdensity=orientationdensity, num=num)
        observations = []
        for pose, config in zip(poses, configs):
            self.calibplanner.moveTo(config)
            # wait for vibration deminishing...
            time.sleep(3.0)
            Tpattern_in_camera = self.capture_corners()
            if Tpattern_in_camera is None:
                continue
            d = dict()
            d['Tpatternincamera'] = Tpattern_in_camera
            d['Tmanipinbase'] = self.calibplanner.vmodel.manip.GetTransform()
            d['config'] = config
            d['pose'] = pose
            observations.append(d)
        return observations


    def capture_corners(self, ):
        checkerboard = (6, 9)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp3d = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
        objp3d[0,:,:2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

        with self.image_lock:
            ret, corners = cv2.findCirclesGrid(
                                self.gray_image, checkerboard,
                                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_ASYMMETRIC_GRID)
            if not ret:
                return None

            corners2 = cv2.cornerSubPix(self.gray_image, corners, (11, 11), (-1, -1), criteria)

            cv_calib_image = cv2.drawChessboardCorners(self.gray_image, checkerboard, corners2, ret)
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_calib_image, "bgr8"))

        # TODO should i use parameters saved in self.env?
        with self.caminfo_lock:
            camera_matrix = np.array(self.cam_info.P).reshape(3,4)
            success, rotation_vector, translation_vector = cv2.solvePnP(objp3d, corners2, camera_matrix, np.array(self.cam_info.distortion_coeffs), flags=0)
        if not success:
            return None
        T = np.eye(4)
        T[0:3,0;3] = cv2.Rodrigues2(rotation_vector)
        T[0:3,3] = translation_vector

        return T




if __name__ == '__main__':
    rospy.init_node('openrave_calibration_planner')
    # self.viewVisibleConfigurations(poses, configs)
    from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())

