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
from visualization import visualize_hits

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


class State(Enum):
    OBSTACLE = auto()
    NO_OBSTACLE = auto()
    TAKEOFF = auto()
    LANDING = auto()
    MANUAL = auto()
    STOP_NAVIGATION = auto()
    CHANGING_COURSE = auto()
    SCANNING = auto()
    PLANNING = auto()


class Planner:
    pass


class Mapper(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)

        self.resize(700, 500)
        self.setWindowTitle('Mapper')

        self.canvas = MapCanvas(self.updateHover)
        self.canvas.create_native()
        self.canvas.native.setParent(self)

        self.setCentralWidget(self.canvas.native)
        
        self.cf = CrazyflieState()
        self.hit_points = None
        self.position = None
        self.measurement = None
        
        # Create grids each cell represents 0.1m by 0.1m area on the ground
        self.occpancy_grids = np.zeros((500,500)) # 50m by 50m empty map
        # (10 meters, 10 meters) is the origin for mapping
        self.position_offset_meters =  10 # to be added to UAV current position
        
        self.hover = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0, 'height': 0.3}
        
        self.hoverTimer = QtCore.QTimer()
        self.hoverTimer.timeout.connect(self.sendHoverCommand)
        self.hoverTimer.setInterval(100)
        self.hoverTimer.start()
        
    def sendHoverCommand(self):
        self.cf.cf.commander.send_hover_setpoint(
        self.hover['x'], self.hover['y'], self.hover['yaw'],
        self.hover['height'])
        self.map_environment()

    def updateHover(self, k, v):
        if (k != 'height'):
            self.hover[k] = v * SPEED_FACTOR
        else:
            self.hover[k] += v
    
    # https://github.com/bitcraze/crazyflie-lib-python/blob/master/examples/multiranger_pointcloud.py
    def rot(self, roll, pitch, yaw, origin, point):
        cosr = math.cos(math.radians(roll))
        cosp = math.cos(math.radians(pitch))
        cosy = math.cos(math.radians(yaw))

        sinr = math.sin(math.radians(roll))
        sinp = math.sin(math.radians(pitch))
        siny = math.sin(math.radians(yaw))

        roty = np.array([[cosy, -siny, 0],
                         [siny, cosy, 0],
                         [0, 0,    1]])

        rotp = np.array([[cosp, 0, sinp],
                         [0, 1, 0],
                         [-sinp, 0, cosp]])

        rotr = np.array([[1, 0,   0],
                         [0, cosr, -sinr],
                         [0, sinr,  cosr]])

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
    
    def update_occupancy_log_odds(self, grid_occupancies):
        for index in grid_occupancies:
            self.occpancy_grids[index[0], index[1]] = grid_occupancies[index]
    
    def convert_to_grid_occupancies(self, hit_point):
        grid_occupancy = {} # value: 0 == not_occupied, 1 == occupied
        cur_row, cur_col = self.get_grid_index_for_position(
            self.offset_position[0], self.offset_position[1])
        hit_row, hit_col = self.get_grid_index_for_position(
            hit_point[0], hit_point[1])
        cells = list(bresenham(cur_row, cur_col, hit_row, hit_col))
        # print("Hit grid: ", hit_row, hit_col)
        # print(cells)
        grid_occupancy[(hit_row, hit_col)] = 1
        for cell in cells[1:-1]:
            grid_occupancy[cell] = 0
            
        return grid_occupancy
    
    def update_occupancy_grids(self, data):
        # multiply the position cooridates by 10 and cast it 
        # to int we can get the grid number
        for hit_point in data:
            grid_occupancy = self.convert_to_grid_occupancies(hit_point)
            self.update_occupancy_log_odds(grid_occupancy)
            
    def map_environment(self):
        if self.cf.measurement is None:
            pass
        else:
            self.offset_position = [self.cf.position[0] + self.position_offset_meters, 
                                    self.cf.position[1] + self.position_offset_meters, 
                                    self.cf.position[2]]
            self.canvas.set_position(self.offset_position)
            data = self.rotate_and_create_points(self.cf.measurement)
            self.update_occupancy_grids(data)
            self.canvas.set_measurement(data)


class MapCanvas(scene.SceneCanvas):
    def __init__(self, keyupdateCB):
        scene.SceneCanvas.__init__(self, keys=None)
        self.size = 800, 600
        self.unfreeze()
        self.view = self.central_widget.add_view()
        self.view.bgcolor = '#ffffff'
        self.view.camera = TurntableCamera(
            fov=20.0, distance=10.0, up='+z', center=(10.0, 10.0, 0.0))
        self.last_pos = [0, 0, 0]
        self.pos_markers = visuals.Markers()
        self.meas_markers = visuals.Markers()
        self.pos_data = np.array([0, 0, 0], ndmin=2)
        self.meas_data = np.array([0, 0, 0], ndmin=2)
        self.lines = []

        self.view.add(self.pos_markers)
        self.view.add(self.meas_markers)
        for i in range(6):
            line = visuals.Line()
            self.lines.append(line)
            self.view.add(line) 
            
        self.keyCB = keyupdateCB
        
        self.freeze()

        scene.visuals.XYZAxis(parent=self.view.scene) 

    def set_position(self, pos):
        self.last_pos = pos
        if (PLOT_CF):
            self.pos_data = np.append(self.pos_data, [pos], axis=0)
            self.pos_markers.set_data(self.pos_data, face_color='red', size=5)

    def set_measurement(self, data):
        o = self.last_pos
        for i in range(6):
            if (i < len(data)):
                o = self.last_pos
                self.lines[i].set_data(np.array([o, data[i]]))
            else:
                self.lines[i].set_data(np.array([o, o]))

        if (len(data) > 0):
            self.meas_data = np.append(self.meas_data, data, axis=0)
        self.meas_markers.set_data(self.meas_data, face_color='blue', size=5)
        
    def on_key_press(self, event):
        if (not event.native.isAutoRepeat()):
            if (event.native.key() == QtCore.Qt.Key_Left):
                self.keyCB('y', 1)
            if (event.native.key() == QtCore.Qt.Key_Right):
                self.keyCB('y', -1)
            if (event.native.key() == QtCore.Qt.Key_Up):
                self.keyCB('x', 1)
            if (event.native.key() == QtCore.Qt.Key_Down):
                self.keyCB('x', -1)
            if (event.native.key() == QtCore.Qt.Key_A):
                self.keyCB('yaw', -70)
            if (event.native.key() == QtCore.Qt.Key_D):
                self.keyCB('yaw', 70)
            if (event.native.key() == QtCore.Qt.Key_Z):
                self.keyCB('yaw', -200)
            if (event.native.key() == QtCore.Qt.Key_X):
                self.keyCB('yaw', 200)
            if (event.native.key() == QtCore.Qt.Key_W):
                self.keyCB('height', 0.1)
            if (event.native.key() == QtCore.Qt.Key_S):
                self.keyCB('height', -0.1)

    def on_key_release(self, event):
        if (not event.native.isAutoRepeat()):
            if (event.native.key() == QtCore.Qt.Key_Left):
                self.keyCB('y', 0)
            if (event.native.key() == QtCore.Qt.Key_Right):
                self.keyCB('y', 0)
            if (event.native.key() == QtCore.Qt.Key_Up):
                self.keyCB('x', 0)
            if (event.native.key() == QtCore.Qt.Key_Down):
                self.keyCB('x', 0)
            if (event.native.key() == QtCore.Qt.Key_A):
                self.keyCB('yaw', 0)
            if (event.native.key() == QtCore.Qt.Key_D):
                self.keyCB('yaw', 0)
            if (event.native.key() == QtCore.Qt.Key_W):
                self.keyCB('height', 0)
            if (event.native.key() == QtCore.Qt.Key_S):
                self.keyCB('height', 0)
            if (event.native.key() == QtCore.Qt.Key_Z):
                self.keyCB('yaw', 0)
            if (event.native.key() == QtCore.Qt.Key_X):
                self.keyCB('yaw', 0)

class Navigator():
    def __init__(self):
        self.VELOCITY = 0.35
        self.TARGET_ALTITUDE = 0.3
        self.state = State.MANUAL
        self.trajectory = []
        self.mc = None
        self.map = Mapper()

    def start_navigation(self):
        keep_flying = True
        while keep_flying:
            if self.cf.measurement is None:
                continue
            else:
                # print(self.cf.measurement)
                # print(self.state)
                if self.state == State.CHANGING_COURSE:
                    continue
                if self.is_close(self.cf.measurement['left']) and self.is_close(self.cf.measurement['right']):
                    self.state = State.STOP_NAVIGATION
                if self.state == State.MANUAL:
                    self.takeoff()
                if self.state == State.TAKEOFF:
                    if self.cf.measurement['down'] > 0.95 * self.TARGET_ALTITUDE:
                        self.state = State.SCANNING
                if self.state == State.SCANNING:
                    hit_points = self.scan_surroundings()
                    self.map.update_hit_points(hit_points, self.cf.position)
                    self.state = State.PLANNING
                if self.state == State.PLANNING:
                    self.travel_to_next_location()
                    self.state = State.SCANNING
                if self.state == State.NO_OBSTACLE:
                    if self.is_close(self.cf.measurement['front']):
                        self.state = State.OBSTACLE
                    elif self.is_close(self.cf.measurement['left']) and self.is_close(self.cf.measurement['right']):
                        self.state = State.STOP_NAVIGATION
                    else:
                        self.start_linear_motion()
                if self.state == State.OBSTACLE:
                    self.change_course(
                        self.cf.measurement['front'], self.cf.measurement['left'],
                        self.cf.measurement['right'], self.cf.measurement['back'])
                    self.state = State.NO_OBSTACLE
                if self.state == State.STOP_NAVIGATION:
                    keep_flying = False

                time.sleep(0.01)

        self.land()

    def travel_to_next_location(self):
        # mc.move_distance(-0.2, 0.0, 0.3, velocity=0.8)
        pass
    
    def scan_surroundings(self):
        hit_points = {}
        front_offset = 90
        right_offset = 0
        back_offset = 270
        left_offset = 180
        for i in range(90):
            self.mc.turn_left(1)
            hit_points[front_offset + i] = self.cf.measurement['front']
            hit_points[right_offset + i] = self.cf.measurement['right']
            hit_points[back_offset + i] = self.cf.measurement['back']
            hit_points[left_offset + i] = self.cf.measurement['left']

        return hit_points

    def takeoff(self):
        self.state = State.TAKEOFF
        self.mc = self.cf.get_motion_commander()
        self.mc.take_off(self.TARGET_ALTITUDE)

    def land(self):
        print("Landing...")
        self.mc.land()

    def is_close(self, range):
        MIN_DISTANCE = 0.6  # m
        if range is None:
            return False
        else:
            return range < MIN_DISTANCE

    def change_course(self, front, left, right, back):
        self.state = State.CHANGING_COURSE
        print("change_course called... front, left, right, back:",
              front, left, right, back)
        if not self.is_close(left):
            print("Attempting to turn left...")
            self.mc.turn_left(90)
            self.state = State.NO_OBSTACLE
        elif not self.is_close(right):
            print("Attempting to turn right...")
            self.mc.turn_right(90)
            self.state = State.NO_OBSTACLE
        else:
            self.state = State.STOP_NAVIGATION

    def start_linear_motion(self):
        self.mc.start_linear_motion(self.VELOCITY, 0, 0)


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
            data['stateEstimate.x'],
            data['stateEstimate.y'],
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
    appQt = QtWidgets.QApplication(sys.argv)
    win = Mapper()
    win.show()
    appQt.exec_()
    # nav = Navigator()
    # nav.start_navigation()