import logging
import sys
import time
import math

import numpy as np
import matplotlib.pyplot as plt
from bresenham import bresenham

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils.multiranger import Multiranger
from cflib.crazyflie.log import LogConfig

from vispy import scene
from vispy.scene import visuals
from vispy.scene.cameras import TurntableCamera

from enum import Enum, auto

try:
    from sip import setapi
    setapi('QVariant', 2)
    setapi('QString', 2)
except ImportError:
    pass

from PyQt5 import QtCore, QtWidgets

URI = 'radio://0/80/2M'

if len(sys.argv) > 1:
    URI = sys.argv[1]

# Enable plotting of Crazyflie
PLOT_CF = False
# Enable plotting of down sensor
PLOT_SENSOR_DOWN = False
# Enable plotting of up sensor
PLOT_SENSOR_UP = False
# Set the sensor threashold (in mm)
SENSOR_TH = 2000
# Set the speed factor for moving and rotating
SPEED_FACTOR = 0.3

# Only output errors from the logging framework
logging.basicConfig(level=logging.INFO)

np.set_printoptions(threshold=sys.maxsize)


class State(Enum):
    TAKEOFF = auto()
    MANUAL = auto()
    STOP_NAVIGATION = auto()
    PLANNING = auto()
    NAVIGATE = auto()


class Mapper:
    def __init__(self):
        self.cf = CrazyflieState()
        self.hit_points = None
        self.position = None
        self.measurement = None

        # Create grids each cell represents 0.1m by 0.1m area on the ground
        # Binary occupancy grid (occupied == 1, un-occupied == 0)
        self.occpancy_grid = np.ones((500, 500))
        # Log-odds occupancy grid initialization
        self.PRIOR_PROB_OCCUPIED = 0.5
        self.OCCUPIED_PROBABILITY = 0.90
        self.occpancy_grid_log_odds = self.initialize_occupancy_grid_log_odds()
        # (10 meters, 10 meters) is the origin for mapping
        self.position_offset_meters = 25  # to be added to UAV current position

    def get_occupancy_map(self):
        # print(np.unique(self.occpancy_grid_log_odds))
        return self.log_odds_to_probability(self.occpancy_grid_log_odds)

    def visualize_occupancy_probabilities(self):
        plt.imshow(self.log_odds_to_probability(self.occpancy_grid_log_odds))
        plt.show()

    def initialize_occupancy_grid_log_odds(self):
        self.occpancy_grid_log_odds = np.ones(
            (500, 500)) * self.PRIOR_PROB_OCCUPIED  # 50m by 50m empty map
        self.occpancy_grid_log_odds = np.log(self.occpancy_grid_log_odds /
                                             (1 - self.PRIOR_PROB_OCCUPIED))

        return self.occpancy_grid_log_odds

    # https://github.com/bitcraze/crazyflie-lib-python/blob/master/examples/multiranger_pointcloud.py
    def rot(self, roll, pitch, yaw, origin, point):
        cosr = math.cos(math.radians(roll))
        cosp = math.cos(math.radians(pitch))
        cosy = math.cos(math.radians(yaw))

        sinr = math.sin(math.radians(roll))
        sinp = math.sin(math.radians(pitch))
        siny = math.sin(math.radians(yaw))

        roty = np.array([[cosy, -siny, 0], [siny, cosy, 0], [0, 0, 1]])

        rotp = np.array([[cosp, 0, sinp], [0, 1, 0], [-sinp, 0, cosp]])

        rotr = np.array([[1, 0, 0], [0, cosr, -sinr], [0, sinr, cosr]])

        rotFirst = np.dot(rotr, rotp)

        rot = np.array(np.dot(rotFirst, roty))

        tmp = np.subtract(point, origin)
        tmp2 = np.dot(rot, tmp)
        return np.add(tmp2, origin)

    # https://github.com/bitcraze/crazyflie-lib-python/blob/master/examples/multiranger_pointcloud.py
    def rotate_and_create_points(self, m):
        data = []
        o = self.offset_position
        roll = m['roll']
        pitch = -m['pitch']
        yaw = m['yaw']

        if (m['left'] < SENSOR_TH):
            left = [o[0], o[1] + m['left'] / 1000.0, o[2]]
            data.append(self.rot(roll, pitch, yaw, o, left))

        if (m['right'] < SENSOR_TH):
            right = [o[0], o[1] - m['right'] / 1000.0, o[2]]
            data.append(self.rot(roll, pitch, yaw, o, right))

        if (m['front'] < SENSOR_TH):
            front = [o[0] + m['front'] / 1000.0, o[1], o[2]]
            data.append(self.rot(roll, pitch, yaw, o, front))

        if (m['back'] < SENSOR_TH):
            back = [o[0] - m['back'] / 1000.0, o[1], o[2]]
            data.append(self.rot(roll, pitch, yaw, o, back))

        return data

    def get_grid_index_for_position(self, pos_x_meters, pos_y_meters):
        return int(pos_x_meters * 10), int(pos_y_meters * 10)

    def log_odds_update(self, l_t_1, p):
        return l_t_1 + np.log(p / (1 - p))

    def update_occupancy_grid_binary(self, grid_occupancies):
        for index in grid_occupancies:
            self.occpancy_grid[index[0], index[1]] = grid_occupancies[index]

    def update_occupancy_grid_log_odds(self, grid_occupancies):
        for index in grid_occupancies:
            l_t_1 = self.occpancy_grid_log_odds[index[0], index[1]]
            if grid_occupancies[index] == 1:
                self.occpancy_grid_log_odds[
                    index[0], index[1]] = \
                        self.log_odds_update(l_t_1, self.OCCUPIED_PROBABILITY)
            else:
                unoccupied_probability = 1 - self.OCCUPIED_PROBABILITY
                self.occpancy_grid_log_odds[
                    index[0], index[1]] = \
                        self.log_odds_update(l_t_1, unoccupied_probability)

    def convert_to_binary_occupancies(self, hit_point):
        grid_occupancy = {}  # value: 0 == not_occupied, 1 == occupied
        cur_row, cur_col = self.get_grid_index_for_position(
            self.offset_position[0], self.offset_position[1])
        hit_row, hit_col = self.get_grid_index_for_position(
            hit_point[0], hit_point[1])
        cells = list(bresenham(cur_row, cur_col, hit_row, hit_col))
        # occupied cells == 1
        grid_occupancy[(hit_row, hit_col)] = 1
        for cell in cells[1:-1]:
            # un-occupied cells == 0
            grid_occupancy[cell] = 0

        return grid_occupancy

    def update_occupancy_grids(self, data):
        # multiply the position cooridates by 10 and cast
        # it to int we can get the grid number
        for hit_point in data:
            binary_grid_occupancy = self.convert_to_binary_occupancies(
                hit_point)
            self.update_occupancy_grid_binary(binary_grid_occupancy)
            self.update_occupancy_grid_log_odds(binary_grid_occupancy)

    def log_odds_to_probability(self, occupancy_grid):
        probs = lambda l: 1 - (1 / (1 + np.exp(l)))
        probs_vec = np.vectorize(probs)
        return probs_vec(occupancy_grid)

    def map_environment(self):
        if self.cf.measurement is None:
            pass
        else:
            self.offset_position = [
                self.cf.position[0] + self.position_offset_meters,
                self.cf.position[1] + self.position_offset_meters,
                self.cf.position[2]
            ]
            data = self.rotate_and_create_points(self.cf.measurement)
            self.update_occupancy_grids(data)


class Navigator():
    def __init__(self):
        self.VELOCITY_X = 0.2
        self.VELOCITY_Y = 0.1
        self.TARGET_ALTITUDE = 0.2
        self.state = State.MANUAL
        self.trajectory = []
        self.waypoints = []
        # vx, vy, yawrate, zdistance
        self.hover_setpoints = [0.2, 0, 0, 0.3]  # control
        self.mc = None
        self.mapper = Mapper()

    def start_navigation(self):
        keep_flying = True
        while keep_flying:
            if self.mapper.cf.measurement is None:
                continue
            else:
                self.mapper.map_environment()
                print(self.state)
                if self.state == State.MANUAL:
                    self.takeoff()
                if self.state == State.TAKEOFF:
                    if self.mapper.cf.measurement[
                            'down'] / 1000.0 > 0.95 * self.TARGET_ALTITUDE:
                        self.state = State.PLANNING
                if self.state == State.PLANNING:
                    velocity_y = 0.0
                    left_close = self.is_close(
                        self.mapper.cf.measurement['left'] / 1000.0)
                    right_close = self.is_close(
                        self.mapper.cf.measurement['right'] / 1000.0)
                    front_close = self.is_close(
                        self.mapper.cf.measurement['front'] / 1000.0)
                    if left_close:
                        velocity_y -= self.VELOCITY_Y
                    if right_close:
                        velocity_y += self.VELOCITY_Y
                    print(self.mapper.cf.measurement['front'] / 1000.0)
                    if left_close and right_close:
                        self.state = State.STOP_NAVIGATION
                        keep_flying = False
                    if front_close:
                        print("Obstacle!")
                        if not left_close:
                            print("Turning left")
                            self.mc.turn_left(90)
                    print("velocities: ", self.VELOCITY_X, velocity_y, 0)
                    self.mc.start_linear_motion(self.VELOCITY_X, velocity_y, 0)
                if self.state == State.STOP_NAVIGATION:
                    keep_flying = False
                    self.land()

                time.sleep(0.1)

    def takeoff(self):
        self.state = State.TAKEOFF
        self.mc = self.mapper.cf.get_motion_commander()
        self.mc.take_off(self.TARGET_ALTITUDE)

    def land(self):
        print("Landing...")
        self.mc.land()
        occupancy_probabilities = self.mapper.get_occupancy_map()
        plt.imshow(occupancy_probabilities)
        plt.axis("equal")
        # plt.grid(True)
        # plt.pause(0.000001)
        plt.show()

    def send_hover_setpoint(self):
        vx, vy, yawrate = self.hover_setpoint
        self.mapper.cf.cf.commander.send_hover_setpoint(
            vx, vy, yawrate, self.TARGET_ALTITUDE)

    def is_close(self, distance):
        MIN_DISTANCE = 0.4  # m
        if distance is None:
            return False
        else:
            return distance < MIN_DISTANCE

    def plan_path(self):
        # plan a path form current location to farthest
        # possible point using the occupancy map
        # time.sleep(1)
        # self.mapper.visualize_occupancy_probabilities()
        prm = ProbablisticRoadMap(self.mapper.get_occupancy_map())
        self.waypoints = prm.get_way_points()
        self.state == State.NAVIGATE

    def horizon_path_planner(self):
        print("horizon path planner called...")
        # time.sleep(1)
        self.hover_setpoint = self.hpp.get_hover_setpoints()
        self.state = State.NAVIGATE

    def navigate_trajectory(self):
        print("Navigating a trajectory...")
        # use cf.commander.send_velocity_world_setpoint(self, vx, vy, vz, yawrate)
        # time.sleep(2)
        # when we run out of way points, we need to plan a path
        self.send_hover_setpoint()
        self.state = State.PLANNING


class Control(Enum):
    FORWARD = auto()
    LEFT_TURN = auto()
    RIGHT_TURN = auto()
    REVERSE = auto()
    LAND = auto()


class CrazyflieState:
    def __init__(self):
        cflib.crtp.init_drivers(enable_debug_driver=False)
        self.cf = Crazyflie(ro_cache=None, rw_cache='cache')
        # self.cf = SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache'))

        # Connect callbacks from the Crazyflie API
        self.cf.connected.add_callback(self.connected)
        self.cf.disconnected.add_callback(self.disconnected)

        # Connect to the Crazyflie
        self.cf.open_link(URI)
        self.measurement = None
        self.position = None

    def get_motion_commander(self):
        return MotionCommander(self.cf)

    def connected(self, URI):
        print('We are now connected to {}'.format(URI))

        # The definition of the logconfig can be made before connecting
        lpos = LogConfig(name='Position', period_in_ms=100)
        lpos.add_variable('stateEstimate.x')
        lpos.add_variable('stateEstimate.y')
        lpos.add_variable('stateEstimate.z')

        try:
            self.cf.log.add_config(lpos)
            lpos.data_received_cb.add_callback(self.pos_data)
            lpos.start()
        except KeyError as e:
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Position log config, bad configuration.')

        lmeas = LogConfig(name='Meas', period_in_ms=100)
        lmeas.add_variable('range.front')
        lmeas.add_variable('range.back')
        lmeas.add_variable('range.up')
        lmeas.add_variable('range.left')
        lmeas.add_variable('range.right')
        lmeas.add_variable('range.zrange')
        lmeas.add_variable('stabilizer.roll')
        lmeas.add_variable('stabilizer.pitch')
        lmeas.add_variable('stabilizer.yaw')

        try:
            self.cf.log.add_config(lmeas)
            lmeas.data_received_cb.add_callback(self.meas_data)
            lmeas.start()
        except KeyError as e:
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Measurement log config, bad configuration.')

    def pos_data(self, timestamp, data, logconf):
        position = [
            data['stateEstimate.x'], data['stateEstimate.y'],
            data['stateEstimate.z']
        ]
        self.position = position

    def _convert_log_to_distance(self, data):
        if data >= 8000:
            return None
        else:
            return data / 1000.0

    def meas_data(self, timestamp, data, logconf):
        measurement = {
            'roll': data['stabilizer.roll'],
            'pitch': data['stabilizer.pitch'],
            'yaw': data['stabilizer.yaw'],
            'front': data['range.front'],
            'back': data['range.back'],
            'up': data['range.up'],
            'down': data['range.zrange'],
            'left': data['range.left'],
            'right': data['range.right']
        }
        self.measurement = measurement

    def disconnected(self, URI):
        print('Disconnected')


if __name__ == '__main__':
    nav = Navigator()
    nav.start_navigation()