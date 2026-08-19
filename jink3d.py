#! /usr/bin/env python

'''Stereography window uses pyqtgraph and opencv to mark features from
two camera views and image the three-dimensional reconstruction of
their paths simultaneously.

'''

import numpy as np
import cv2 as cv
import configparser
import os
import sys

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtGui
from PyQt5 import QtWidgets, QtCore
import pyqtgraph.console

from scipy.interpolate import InterpolatedUnivariateSpline


def cm_soapbubble(x, y):
    '''soapbubble colors for complex magnitudes and angles.

    x = intensity, lightness, 0 -- 1,

    y = radians

    '''
    r = np.clip(-57.4 * np.sin(y + 4.5) + x * 265.1 + -20.3, 0, 255)
    g = np.clip(20.6 * np.sin(y + -1.5) + x * 253.5 + -14.3, 0, 255)
    b = np.clip(57.1 * np.sin(y + 3.2) + x * 234.6 + -4.3, 0, 255)
    return np.stack([int(r), int(g), int(b)])


# make a ring of colors with the same darkness
colors = [cm_soapbubble(.5, y) for y in np.linspace(0, 2 * np.pi, 9, endpoint=False)]
# mix the colors so near ones aren't adjacent
mix = ((np.arange(9) * 2) % 9)
# start with blue and green
mix = np.roll(mix[::-1], 4)
colors = [cm_soapbubble(.6, y) for y in mix / 9. * 2 * np.pi]


def _opencv_image_points(points, name='image points'):
    """Return OpenCV 2D calibration points as float32 (N, 1, 2)."""
    points = np.asarray(points)
    original_shape = points.shape
    if points.ndim == 2 and points.shape[1] == 2:
        pass
    elif points.ndim == 3 and points.shape[1:] == (1, 2):
        pass
    else:
        raise ValueError(
            f'{name} must have shape (N, 2) or (N, 1, 2); got {original_shape}'
        )
    return np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)


def _opencv_corner_ids(ids, name='corner IDs'):
    """Return OpenCV corner IDs as int32 (N, 1)."""
    ids = np.asarray(ids)
    original_shape = ids.shape
    if ids.ndim == 1:
        pass
    elif ids.ndim == 2 and ids.shape[1] == 1:
        pass
    else:
        raise ValueError(f'{name} must have shape (N,) or (N, 1); got {original_shape}')
    return np.asarray(ids, dtype=np.int32).reshape(-1, 1)


class TableSizeSelector(QtWidgets.QWidget):
    def __init__(self, st_win=None, parent=None):
        super().__init__()
        self.current_row = 0
        self.current_col = 0
        self.initUI()

        self.st_win = st_win

    def initUI(self):
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)
        layout.setSpacing(0)
        layout.setContentsMargins(1, 1, 1, 1)

        self.buttons = []
        for row in range(16):
            button_row = []
            for col in range(16):
                button = QtWidgets.QPushButton(self)
                button.setFixedSize(20, 20)
                button.setStyleSheet("""
                    QPushButton {
                        border: 1px solid #ccc;
                        background-color: white;
                    }
                    QPushButton:hover {
                        background-color: #e6e6e6;
                    }
                """)
                button.enterEvent = lambda event, r=row, c=col: self.highlight_cells(r, c)
                button.clicked.connect(lambda _, r=row, c=col: self.on_click(r, c))
                layout.addWidget(button, row, col)
                button_row.append(button)
            self.buttons.append(button_row)

        self.size_label = QtWidgets.QLabel("0 x 0")
        self.size_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.size_label, 8, 0, 1, 8)

        self.setMouseTracking(True)
        self.leaveEvent = lambda event: self.clear_highlight()

    def highlight_cells(self, row, col):
        self.current_row = row
        self.current_col = col
        for r in range(16):
            for c in range(16):
                if r <= row and c <= col:
                    self.buttons[r][c].setStyleSheet("""
                        QPushButton {
                            border: 1px solid #ccc;
                            background-color: #e6e6e6;
                        }
                    """)
                else:
                    self.buttons[r][c].setStyleSheet("""
                        QPushButton {
                            border: 1px solid #ccc;
                            background-color: white;
                        }
                    """)
        self.size_label.setText(f"{row + 1} x {col + 1}")

    def clear_highlight(self):
        for row in self.buttons:
            for button in row:
                button.setStyleSheet("""
                    QPushButton {
                        border: 1px solid #ccc;
                        background-color: white;
                    }
                    QPushButton:hover {
                        background-color: #e6e6e6;
                    }
                """)
        self.size_label.setText("0 x 0")

    def on_click(self, row, col):
        print(f"Selected board size: {row + 1}x{col + 1}")
        # print(f"{dir(self.parent())}")
        self.parent().close()
        self.st_win.set_chessboard_size(row + 1, col + 1)


###############
# TDframe
###############
# tdddd

class TDframe():
    '''The three dimensional representation trajectories from image data,
    including a packing box for the plots, 3d data for plotting, and
    links to the Imframes that have 2d data and projection matrixes
    for calculating 3d data.

    '''

    def __init__(self, box, ims, num_markers=9, num_frames=1):
        # pyqtgraph's opengl 3d viewer
        self.view = gl.GLViewWidget()
        self.lines = []
        self.points = []
        self.num_markers = num_markers
        self.num_frames = num_frames

        self.view.setBackgroundColor((0, 0, 0))
        # self.view.setWindowTitle('pyqtgraph example: GLLinePlotItem')
        self.view.opts['distance'] = 60
        self.view.opts['elevation'] = 15
        self.view.opts['azimuth'] = -90

        # grid
        self.gz = gl.GLGridItem(color=pg.mkColor([76, 114, 175, 255]))
        self.gz.translate(0, 0, -10)
        self.view.addItem(self.gz)

        self.ax = gl.GLAxisItem()
        self.view.addItem(self.ax)

        self.curr_pan = np.array([-1, 0, 0, 0])
        self.focus = np.array([0, 0, 0])

        self.c1 = gl.GLTextItem()
        self.c1.setData(text='C1')
        self.view.addItem(self.c1)

        self.c2 = gl.GLTextItem()
        self.c2.setData(text='C2')
        self.c2.setVisible(False)
        self.view.addItem(self.c2)

        for marker_ind in range(self.num_markers):
            line = gl.GLLinePlotItem(pos=np.array([[0, 0, 0.], [1, 1, 1.]]),
                                     color=pg.glColor(colors[marker_ind]), width=2., antialias=True)
            line.hide()
            self.lines.append(line)
            self.view.addItem(line)

            pt = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0.]]),
                                      color=pg.glColor(colors[marker_ind]), size=10.)
            pt.hide()
            self.points.append(pt)
            self.view.addItem(pt)

        self.view.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        box.addWidget(self.view)

        self.num_markers = num_markers
        self.num_frames = num_frames
        self.data = np.zeros((num_markers, 3, 1))
        self.ims = ims

        # self.nmarkers = gl.GLScatterPlotItem(pos=np.array([[0,0,0.]]),
        #                                      color=pg.glColor(colors[marker_ind]), size=10.)
        self.nmarkers = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0.]]),
                                             color=pg.glColor(colors[8]), size=5.)

        self.view.addItem(self.nmarkers)
        self.nmarkers.hide()  # initially invisible

        # self.pbrot, is the rotation matrix to make the plumbline,
        self.pb = np.array([0, 0, -1])
        # default np.eye(3) does nothing
        self.pbrot = np.eye(3)

        self.got_pb = None

        # calibration
        self.got_cal = None

    def set_data(self, data=None, num_markers=None, num_frames=None):
        '''Set the data, or create a new data structure when the number of
        frames or markers changes, such as loading a new avi

        '''
        # print('set data')
        if data is not None:
            self.data = data
            self.num_markers = data.shape[0]
            self.num_frames = data.shape[-1]
        else:
            self.num_markers = num_markers if num_markers else self.ims[0].num_markers
            self.num_frames = num_frames if num_frames else self.ims[0].num_frames
            self.data = np.zeros((self.num_markers, 3, self.num_frames))

        self.nmarkers.hide()

    def update_data(self, marker_ind):
        '''Recalculate all the 3d data for a marker, from the Imframes, then
        adjust the plumbline direction to downwards.

        '''
        # print('update data ', marker_ind)
        marked = [np.where(im.data[marker_ind, -1])[0] for im in self.ims]
        if marked[0].size > 0 and marked[1].size > 0:
            valid_min = max([m.min() for m in marked])
            valid_max = min([m.max() for m in marked])
            valid_inds = np.arange(valid_min, valid_max + 1, dtype='int')
        else:
            valid_inds = []

        if len(valid_inds) > 0:
            # get undistorted points
            self.pts0 = _opencv_image_points(
                self.ims[0].data[marker_ind, :2, valid_inds], 'left marker image points'
            )
            self.upts0 = cv.undistortPoints(self.pts0, self.ims[0].mtx, self.ims[0].dist,
                                            np.eye(3), self.ims[0].mtx)

            self.pts1 = _opencv_image_points(
                self.ims[1].data[marker_ind, :2, valid_inds], 'right marker image points'
            )
            self.upts1 = cv.undistortPoints(self.pts1, self.ims[1].mtx, self.ims[1].dist,
                                            np.eye(3), self.ims[1].mtx)

            upts0_xy = np.asarray(self.upts0).reshape(-1, 2)
            upts1_xy = np.asarray(self.upts1).reshape(-1, 2)
            if upts0_xy.shape != upts1_xy.shape:
                raise ValueError(
                    f'Undistorted camera point shapes do not match: '
                    f'{upts0_xy.shape} and {upts1_xy.shape}'
                )
            st_pts = cv.triangulatePoints(self.ims[0].proj, self.ims[1].proj,
                                          upts0_xy.T, upts1_xy.T)

            # normalize homogenous coordinates
            st_pts /= st_pts[-1]
            self.st = st_pts

            # swap y and z so z is up, y is dist from camera
            # and homogenous 4 to euclidian 3
            rmap = np.array([[1, 0, 0, 0],
                             [0, 0, 1, 0],
                             [0, -1, 0, 0]])

            # adjust downwards to plumbline
            st_pts = self.pbrot @ rmap @ st_pts

            self.st_pts = st_pts

            self.data[marker_ind, :, valid_inds] = st_pts.T

    def get_valid_inds(self, marker_ind):
        '''Find the indexes where between which we can interpolate a marker
        position

        '''
        marked = [np.where(im.data[marker_ind, -1])[0] for im in self.ims]
        if marked[0].size > 0 and marked[1].size > 0:
            valid_min = max([m.min() for m in marked])
            valid_max = min([m.max() for m in marked])
            valid_inds = np.arange(valid_min, valid_max + 1, dtype='int')
        else:
            valid_inds = []

        return valid_inds

    def pan(self, marker_ind):
        '''Try to change the view focus

        '''
        # undo the old pan
        back = -self.focus
        self.view.pan(*back)
        self.focus[:] = 0, 0, 0

        # -1 indicates pan to the origin (camera 0)
        if marker_ind == -1:
            return

        else:
            valid_inds = self.get_valid_inds(marker_ind)
            # if the nmarkers are showing (like in a calibration
            # frame) pan to them instead of marker 1
            if marker_ind == 0 and self.nmarkers.visible():
                x, y, z = self.nmarkers.pos.mean(0)

            # other markers, if there are valid inds
            elif len(valid_inds) > 0:
                x, y, z = self.data[marker_ind, :, valid_inds].mean(0)

            else:
                x, y, z = 0., 0., 0.

            self.view.pan(x, y, z)
            self.focus[:] = x, y, z

    def show_lines(self):
        '''Checks if we should draw each line (if it has some valid inds),
        then draws it based on existing, updated data, otherwise hides
        it.

        '''
        # print('show lines')
        xmax, ymax, zmax = 20, 20, 20
        for marker_ind in np.arange(self.num_markers):
            valid_inds = self.get_valid_inds(marker_ind)

            if len(valid_inds) > 0:
                line_data = self.data[marker_ind, :, valid_inds]
                self.lines[marker_ind].setData(pos=line_data)
                self.lines[marker_ind].show()

                xm, ym, zm = self.data[marker_ind, :, valid_inds].max(0)
                if xm > xmax: xmax = xm
                if ym > ymax: ymax = ym
                if zm > zmax: zmax = zm

            else:
                self.lines[marker_ind].hide()

        # does this frame have visible markers?
        if self.nmarkers.visible():
            xm, ym, zm = self.nmarkers.pos.max(0)
            if xm > xmax: xmax = xm
            if ym > ymax: ymax = ym
            if zm > zmax: zmax = zm

        # resizes the grid
        flr = 2 * max(xmax, ymax)
        spc = 10 ** (np.floor(np.log10(flr)) - 1)
        self.gz.setSize(flr, flr, 1)
        self.gz.setSpacing(spc, spc)

    def set_frame(self, n):
        '''Checks if we should draw each marker (if the requested frame is in
        valid inds), then draws it, otherwides hides it.

        '''
        # print('set frame ', n)
        for marker_ind in np.arange(self.num_markers):
            valid_inds = self.get_valid_inds(marker_ind)

            if n in valid_inds:
                frame_pt = self.data[marker_ind, :, n]
                self.points[marker_ind].setData(pos=frame_pt)
                self.points[marker_ind].show()
            else:
                self.points[marker_ind].hide()

        # if this is a calibration frame, show the triangulated
        # calibration points in the 3D view.  ChArUco detections may
        # include different corner IDs in the two cameras, so use only
        # corners detected in both views.
        if n in self.ims[0].cal_inds and n in self.ims[1].cal_inds:
            cal_ind0 = self.ims[0].cal_inds.index(n)
            cal_ind1 = self.ims[1].cal_inds.index(n)

            pts0 = _opencv_image_points(
                self.ims[0].image_corners[cal_ind0], 'left calibration image points'
            )
            pts1 = _opencv_image_points(
                self.ims[1].image_corners[cal_ind1], 'right calibration image points'
            )

            ids0 = getattr(self.ims[0], 'image_corner_ids', None)
            ids1 = getattr(self.ims[1], 'image_corner_ids', None)

            # ChArUco: keep only shared corner IDs and sort both cameras
            # into the same ID order before triangulating.
            if ids0 and ids1:
                ids0_frame = _opencv_corner_ids(ids0[cal_ind0], 'left ChArUco IDs').ravel()
                ids1_frame = _opencv_corner_ids(ids1[cal_ind1], 'right ChArUco IDs').ravel()
                if len(ids0_frame) != len(pts0) or len(ids1_frame) != len(pts1):
                    raise ValueError(
                        'ChArUco corner/ID count mismatch: '
                        f'left points {pts0.shape}, IDs {ids0_frame.shape}; '
                        f'right points {pts1.shape}, IDs {ids1_frame.shape}'
                    )
                common_ids = np.intersect1d(ids0_frame, ids1_frame)

                if common_ids.size >= 2:
                    take0 = np.nonzero(np.isin(ids0_frame, common_ids))[0]
                    take1 = np.nonzero(np.isin(ids1_frame, common_ids))[0]

                    take0 = take0[np.argsort(ids0_frame[take0])]
                    take1 = take1[np.argsort(ids1_frame[take1])]

                    pts0 = pts0[take0]
                    pts1 = pts1[take1]
                else:
                    pts0 = np.empty((0, 1, 2), dtype=np.float32)
                    pts1 = np.empty((0, 1, 2), dtype=np.float32)

            # Checkerboard or circle grid: both views should already have the
            # same number of points. If not, do not crash while inspecting.
            if len(pts0) == len(pts1) and len(pts0) >= 2:
                pts0_xy = np.asarray(pts0).reshape(-1, 2)
                pts1_xy = np.asarray(pts1).reshape(-1, 2)
                h_pts = cv.triangulatePoints(self.ims[0].proj, self.ims[1].proj,
                                             pts0_xy.T,
                                             pts1_xy.T)
                # divide out for homogenous coordinates
                h_pts[:3] /= h_pts[3]

                rmap = np.array([[1, 0, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, -1, 0, 0]])

                h_pts = rmap @ h_pts

                self.h_pts = h_pts
                self.nmarkers.setData(pos=h_pts.T)
                self.nmarkers.show()
            else:
                self.nmarkers.hide()
        else:
            self.nmarkers.hide()

        self.show_lines()

    def set_camera_markers(self):
        '''Update the position of the cameras, after we have a T matrix. C1
        remains at the origin.

        '''
        # print('set camera markers')
        rr = -np.dot(self.R.T, self.T[:, 0])
        self.c2.setData(pos=(rr[0], -rr[2], rr[1]))
        self.c2.setVisible(True)

    def get_calibration(self):
        '''Find board corners in frames marked with marker 1, and calculate
        the calibration matrixes.

        '''
        st_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
        st_flags = cv.CALIB_FIX_INTRINSIC

        self.cal_inds = []
        self.board_corners = []
        self.image0_corners = []
        self.image1_corners = []

        # grab the calibration indexes for each camera
        ind_lists = [im.cal_inds for im in self.ims]

        # add only indexes that appear on each list
        for ind in ind_lists[0]:
            if all([ind_list.count(ind) for ind_list in ind_lists]):
                self.cal_inds.append(int(ind))
        print(self.cal_inds)

        # append the board and image corner locations
        for ind in self.cal_inds:
            l_ind = self.ims[0].cal_inds.index(ind)
            r_ind = self.ims[1].cal_inds.index(ind)

            left_ids = getattr(self.ims[0], 'image_corner_ids', None)
            right_ids = getattr(self.ims[1], 'image_corner_ids', None)

            if left_ids and right_ids:
                # ChArUco boards may produce different visible corner sets in
                # the two views. Keep only IDs detected in both cameras.
                l_ids = _opencv_corner_ids(left_ids[l_ind], 'left ChArUco IDs').ravel()
                r_ids = _opencv_corner_ids(right_ids[r_ind], 'right ChArUco IDs').ravel()
                l_points = _opencv_image_points(
                    self.ims[0].image_corners[l_ind], 'left ChArUco image points'
                )
                r_points = _opencv_image_points(
                    self.ims[1].image_corners[r_ind], 'right ChArUco image points'
                )
                if len(l_ids) != len(l_points) or len(r_ids) != len(r_points):
                    raise ValueError(
                        'ChArUco corner/ID count mismatch: '
                        f'left points {l_points.shape}, IDs {l_ids.shape}; '
                        f'right points {r_points.shape}, IDs {r_ids.shape}'
                    )
                common_ids = np.intersect1d(l_ids, r_ids)

                if common_ids.size < 6:
                    print(f"{ind} only {common_ids.size} shared ChArUco corners")
                    continue

                l_take = np.nonzero(np.isin(l_ids, common_ids))[0]
                r_take = np.nonzero(np.isin(r_ids, common_ids))[0]

                # Sort both views by corner ID so object image correspondences match.
                l_take = l_take[np.argsort(l_ids[l_take])]
                r_take = r_take[np.argsort(r_ids[r_take])]

                self.board_corners.append(self.ims[0].board_corners[l_ind][l_take])
                self.image0_corners.append(l_points[l_take])
                self.image1_corners.append(r_points[r_take])
            else:
                board_points = self.ims[0].board_corners[l_ind]
                left_points = _opencv_image_points(
                    self.ims[0].image_corners[l_ind], 'left calibration image points'
                )
                right_points = _opencv_image_points(
                    self.ims[1].image_corners[r_ind], 'right calibration image points'
                )
                if len(board_points) != len(left_points) or len(left_points) != len(right_points):
                    raise ValueError(
                        f'Calibration point count mismatch at frame {ind}: '
                        f'object {np.shape(board_points)}, left {left_points.shape}, '
                        f'right {right_points.shape}'
                    )
                self.board_corners.append(board_points)
                self.image0_corners.append(left_points)
                self.image1_corners.append(right_points)

        if len(self.board_corners) < 3:
            raise ValueError('Need at least 3 stereo calibration frames with shared board points')

        out = cv.stereoCalibrate(self.board_corners,
                                 self.image0_corners, self.image1_corners,
                                 self.ims[0].mtx, self.ims[0].dist,
                                 self.ims[1].mtx, self.ims[1].dist,
                                 (self.ims[0].w, self.ims[0].h),
                                 criteria=st_criteria, flags=st_flags)

        self.rmse, self.l_cm, self.l_dist, self.r_cm, self.r_dist, self.R, self.T, self.E, self.F = out

        # projection matrixes the product of the camera matrix and the rotation and translation matrixes
        self.ims[0].proj = self.ims[0].mtx @ cv.hconcat([np.eye(3), np.zeros((3, 1))])
        self.ims[1].proj = self.ims[1].mtx @ cv.hconcat([self.R, self.T])

        self.got_cal = True

        self.set_camera_markers()

    def _rotation_from_vector_to_down(self, vec):
        '''Return a rotation matrix that maps vec onto the negative z axis.

        This is used for orientation calibrations: plumbline vectors and
        projectile acceleration vectors both define the down direction.
        '''
        vec = np.asarray(vec, dtype=float)
        norm = np.linalg.norm(vec)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError('Orientation vector has zero or invalid length')

        uvec = vec / norm
        target = np.array([0., 0., -1.])
        dot = np.clip(np.dot(uvec, target), -1., 1.)

        # Already aligned with down.
        if np.isclose(dot, 1.):
            return np.eye(3)

        # Exactly opposite down: rotate 180 degrees around any perpendicular axis.
        if np.isclose(dot, -1.):
            axis = np.cross(uvec, np.array([1., 0., 0.]))
            if np.linalg.norm(axis) == 0:
                axis = np.cross(uvec, np.array([0., 1., 0.]))
            axis = axis / np.linalg.norm(axis)
            angle = np.pi
        else:
            axis = np.cross(uvec, target)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(dot)

        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    def get_projectile_orientation(self, marker_ind=0, fps=1.0):
        '''Use the acceleration of a marked projectile to define down.

        The marker should track the center of a freely thrown object across
        frames. The camera geometry must already be calibrated so that 2D
        camera points can be reconstructed in 3D. The fitted second derivative
        gives the gravity vector in calibration-units per second squared.
        '''
        if self.got_cal is None:
            raise ValueError('Need camera geometry before projectile orientation')

        fps = float(fps)
        if fps <= 0:
            raise ValueError('FPS must be positive')

        # Refresh 3D data for this marker in case points were edited after the
        # last reconstruction. Keep the previous orientation because the
        # reconstructed points (and therefore their acceleration) already
        # include it.
        previous_rotation = self.pbrot.copy()
        for im in self.ims:
            im.make_interp(marker_ind)
        self.update_data(marker_ind)

        marked = [np.where(im.data[marker_ind, -1] == 1)[0] for im in self.ims]
        if marked[0].size == 0 or marked[1].size == 0:
            raise ValueError(f'Marker {marker_ind + 1} is not marked in both camera views')

        inds = np.intersect1d(marked[0], marked[1]).astype(int)
        if inds.size < 3:
            raise ValueError('Need at least 3 shared marked frames to estimate acceleration')

        # Combining a scalar marker index with the advanced frame index puts
        # the frame dimension first, so this is already shaped (frames, 3).
        pts = self.data[marker_ind, :, inds]
        finite = np.all(np.isfinite(pts), axis=1)
        inds = inds[finite]
        pts = pts[finite]

        if inds.size < 3:
            raise ValueError('Need at least 3 finite 3D positions to estimate acceleration')

        t = inds.astype(float) / fps
        k = min(3, inds.size - 1)
        if k < 2:
            raise ValueError('Need enough frames for at least a quadratic fit')

        accels = []
        for coord in range(3):
            spline = InterpolatedUnivariateSpline(t, pts[:, coord], k=k)
            accels.append(spline.derivative(2)(t))
        accels = np.vstack(accels).T

        # Avoid the outermost samples for cubic splines, where edge behavior can
        # be unstable. With only three or four samples, use all available data.
        if accels.shape[0] > 4:
            accel = np.nanmean(accels[1:-1], axis=0)
        else:
            accel = np.nanmean(accels, axis=0)

        correction = self._rotation_from_vector_to_down(accel)
        self.pbrot = correction @ previous_rotation
        self.got_pb = True

        # Apply the new coordinate system immediately to every reconstructed
        # marker, rather than waiting for the user to edit each marker again.
        for ind in range(self.num_markers):
            self.update_data(ind)

        mag = np.linalg.norm(accel)
        return (
            f'thrown-object orientation from marker {marker_ind + 1}\n'
            f'frames used: {inds.tolist()}\n'
            f'acceleration vector: {accel}\n'
            f'acceleration magnitude: {mag:.3f} calibration-units/s^2'
        )

    def get_plumbline(self):
        '''Use markers 1 and 2 to calculate a plumbline matrix.

        '''
        # find the frame indexes with markers 1 and 2, in both l and r views
        l1_inds = self.ims[0].data[0, -1] == 1
        l2_inds = self.ims[0].data[1, -1] == 1
        r1_inds = self.ims[1].data[0, -1] == 1
        r2_inds = self.ims[1].data[1, -1] == 1
        inds = np.where(l1_inds & l2_inds & r1_inds & r2_inds)[0]

        if len(inds) > 0:

            previous_rotation = self.pbrot.copy()

            m1 = self.data[0, :, inds].mean(0)
            m2 = self.data[1, :, inds].mean(0)

            # plumbline is the vector pointing toward the true down
            self.pb = m2 - m1

            correction = self._rotation_from_vector_to_down(self.pb)
            self.pbrot = correction @ previous_rotation

            self.got_pb = True
            for marker_ind in range(self.num_markers):
                self.update_data(marker_ind)
            return 'found new plumbline'

        else:
            return "couldn't find plumbline---use marker 1 and 2 in at least one pair of frames"


###############
### Imframe ###
###############
# immmm

class Imframe():
    '''image and data showing a frame of an avi, that we can scale,
    update, add markers to, and adjust a histogram

    '''

    def __init__(self, box, parent, num_markers=9):
        # add the image to whatever box was passed
        self.imview = pg.ImageView()
        self.imhist = self.imview.getHistogramWidget()

        self.randim = np.random.randint(256, size=(640, 480))
        self.imview.setImage(self.randim)
        self.imview.setMinimumHeight(400)
        self.imview.getHistogramWidget().setMaximumWidth(100)
        self.imview.scene.sigMouseMoved.connect(self.mouse_moved)
        self.imview.scene.sigMouseClicked.connect(self.mouse_clicked)
        self.fn = ''

        box.addWidget(self.imview)

        # data
        self.num_frames = 1
        self.num_markers = num_markers
        self.data = np.zeros([self.num_markers, 3, self.num_frames])
        # interp data
        self.null_interp = InterpolatedUnivariateSpline([0, 0.01], [np.nan, np.nan], k=1)
        self.interp = [[self.null_interp for xy in range(2)] for m in range(self.num_markers)]
        # markers for data
        self.markers = [pg.TargetItem() for marker in range(self.num_markers)]
        for marker_ind, marker in enumerate(self.markers):
            marker.setToolTip(str(marker_ind + 1))
            marker.setPen(QtGui.QPen(QtGui.QColor(*colors[marker_ind])))
            # marker.sigPositionChanged.connect(self.marker_moved)
            marker.hide()  # initially invisible
            self.imview.addItem(marker)

        # markers that aren't editable (for calibration)
        self.nmarkers = pg.ScatterPlotItem(symbol='o', setPen=(245, 245, 30, 127))
        self.imview.addItem(self.nmarkers)
        self.nmarkers.clear()  # initially invisible

        self.mousepos = (0, 0)
        self.cal_inds = []

        # for communicating with the main window
        self.parent = parent

    def load_avi(self, fn):
        '''Get an avi file and read and display the first frame'''
        self.fn = fn
        self.cap = cv.VideoCapture(fn)
        self.num_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.frame_ind = 0

        self.cap.set(cv.CAP_PROP_POS_FRAMES, self.frame_ind)
        ret, im = self.cap.read()
        if ret:
            self.image = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
            self.h, self.w = self.image.shape

        self.cal_inds = []

    def load_folder(*self, fn):
        '''Load a folder full of images instead of an avi'''
        self.fn = fn

    def set_frame(self, n, autorange=False, autolevel=False):
        '''Read the image of a new frame, convert to black and white, trigger
        update

        '''
        self.cap.set(cv.CAP_PROP_POS_FRAMES, n)
        ret, im = self.cap.read()
        if ret:
            self.frame_ind = n
            self.image = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
            self.update_im(autorange, autolevel)
            self.show_markers()

    def update_im(self, autorange=False, autolevel=False):
        '''Reset the image after it gets changed

        '''
        self.imview.setImage(self.image.T, autoRange=autorange, autoLevels=autolevel)

    def adjust_levels_quantile(self, qmin=.1, qmax=None):
        '''Change the levels based on the quantile intensity range in the
        current image.

        '''
        if qmax is None:
            qmax = 1. - qmin

        lmin, lmax = np.quantile(self.image, [qmin, qmax])

        self.imview.setLevels(lmin, lmax)

    def get_marker(self, marker_ind):
        '''Get the position of the marker, and interpolate if it's not marked
        directly

        '''
        x, y, marked = self.data[marker_ind, :, self.frame_ind]
        if not marked:
            try:  # this will only work if we are in the domain (interpolating)
                x = self.interp[marker_ind][0](self.frame_ind)
                y = self.interp[marker_ind][1](self.frame_ind)
            except:  # if we are extrapolating, return nan, so we don't draw a marker
                x = np.nan
                y = np.nan
        return x, y, marked

    def set_marker(self, marker_ind, frame_ind=None, pos=None, add=True):
        '''Set a marker position with coordinates, or from the current mouse
        position. Or if add is false, remove the marker. Then update
        the interpolation, and show the new position.

        '''
        if frame_ind is None:
            frame_ind = self.frame_ind

        # for undo
        previous = marker_ind, frame_ind, self.data[marker_ind, :, self.frame_ind].copy()

        if add:
            if pos is not None:
                x, y, s = pos
            else:
                x = self.imview.imageItem.mapFromScene(self.mousepos).x()
                y = self.imview.imageItem.mapFromScene(self.mousepos).y()
                s = 1

            self.data[marker_ind, :, self.frame_ind] = [x, y, s]

        # or remove the marked
        else:
            self.data[marker_ind, -1, self.frame_ind] = 0

        # set undo
        undo_call = self.set_marker, previous
        self.parent.undo_list.append(undo_call)

        # update interp and display
        self.make_interp(marker_ind)
        self.show_markers()

    def show_markers(self):
        '''Show markers that we have data for, or we can interpolate, or else
        hide them

        '''
        for marker_ind in np.arange(self.num_markers):
            x, y, marked = self.get_marker(marker_ind)

            # if we didn't get an xy, turn off the marker
            if np.isnan(x + y):
                self.markers[marker_ind].hide()

            else:
                alpha = 200 if marked else 50
                brush = pg.mkBrush(*colors[marker_ind], alpha)

                self.markers[marker_ind].setBrush(brush)
                self.markers[marker_ind].setPos((x, y))
                self.markers[marker_ind].show()

        # if this is a calibration frame
        if self.frame_ind in self.cal_inds:
            cal_ind = self.cal_inds.index(self.frame_ind)
            points = _opencv_image_points(
                self.image_corners[cal_ind], 'calibration marker display points'
            ).reshape(-1, 2)
            self.nmarkers.setData(pos=points, symbol='o')
            self.nmarkers.setBrush(pg.mkBrush('y'))
            self.nmarkers.show()
        else:
            self.nmarkers.clear()

    def get_data(self, marker_ind, frame_ind):
        '''Get the position of the marker, and interpolate if it's not marked
        directly. Maybe this should replace self.get_marker

        '''
        x, y, marked = self.data[marker_ind, :, frame_ind]
        if not marked:
            try:  # this will only work if we are in the domain (interpolating)
                x = self.interp[marker_ind][0](frame_ind)
                y = self.interp[marker_ind][1](frame_ind)
            except:  # if we are extrapolating, return nan, so we don't draw a marker
                x = np.nan
                y = np.nan
        return x, y, marked

    def set_data(self, data=None, num_markers=None, num_frames=None):
        '''Set all the data, after newly loaded avi, and remake the
        interpolations for each marker

        '''
        if data is not None:
            self.data = data
            self.num_markers = data.shape[0]
            self.num_frames = data.shape[-1]
            # for marker_ind in arange(self.num_markers):
            #     self.make_interp(marker_ind)
        else:
            self.num_markers = num_markers if num_markers else self.num_markers
            self.num_frames = num_frames if num_frames else self.num_frames
            self.data = np.zeros((self.num_markers, 3, self.num_frames))

        for marker_ind in np.arange(self.num_markers):
            self.make_interp(marker_ind)

    def make_interp(self, marker_ind):
        '''Make spline interpolations for x and y positions of the markers
        based on frames that are marked. This requires at least 2
        filled frames, and the spline order will be one less than the
        number of filled frames, up to a cubic spline.

        '''
        num_filled_frames = sum(self.data[marker_ind, -1])
        if num_filled_frames >= 2:
            kval = np.clip(num_filled_frames - 1, 1, 3)

            # get values at marked frames
            ts = np.where(self.data[marker_ind, -1] == 1)
            xs = self.data[marker_ind, 0][ts]
            ys = self.data[marker_ind, 1][ts]

            self.interp[marker_ind][0] = InterpolatedUnivariateSpline(ts[0], xs, k=kval, ext='raise')
            self.interp[marker_ind][1] = InterpolatedUnivariateSpline(ts[0], ys, k=kval, ext='raise')

            knots = self.interp[marker_ind][0].get_knots()
            valid_frames = np.arange(knots.min(), knots.max(), dtype='int')
            self.data[marker_ind, 0, valid_frames] = self.interp[marker_ind][0](valid_frames)
            self.data[marker_ind, 1, valid_frames] = self.interp[marker_ind][1](valid_frames)

        else:
            self.interp[marker_ind] = [self.null_interp, self.null_interp]

    def mouse_moved(self, pos):
        """If the mouse moves in this axis, update current position, so we
        know if a marker is set with the keyboard. A keyboard event
        won't otherwise report the mouse position.

        """
        self.mousepos = pos

    def mouse_clicked(self, pos):
        '''Clicking the mouse, like hitting a number key, will add a marker to
        the current mouse position.

        '''
        self.mcpos = pos

        if pos.button == 2 or (
                pos.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier) == QtCore.Qt.KeyboardModifier.AltModifier:
            add = False
        else:
            add = True

        self.set_marker(self.parent.curr_marker, add=add)

    def get_calibration(self, inds, num_rows=7, num_cols=6, side_len=11.5, use_roi=True):
        """Try to get calibration from marked frames, then set the projection
        matrix.
        """
        self.cal_inds = []
        self.image_corners = []
        self.board_corners = []
        self.image_corner_ids = []

        chess_flags = (
                cv.CALIB_CB_ADAPTIVE_THRESH
                + cv.CALIB_CB_NORMALIZE_IMAGE
                + cv.CALIB_CB_FILTER_QUADS
        )
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # object points in real board coordinates
        board_pts = np.zeros((num_rows * num_cols, 3), np.float32)
        board_pts[:, :2] = np.mgrid[0:num_rows, 0:num_cols].T.reshape(-1, 2)
        board_pts *= side_len

        for ind in inds:
            self.cap.set(cv.CAP_PROP_POS_FRAMES, ind)
            im_found, im = self.cap.read()

            if not im_found or im is None:
                print(f"{ind} no im found")
                continue

            im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

            x_min, y_min = 0, 0
            if use_roi:
                mn, mx = self.imhist.item.getLevels()
                i = np.interp(im, [mn, mx], [0, 255]).astype("uint8")

                # visible range in data coordinates
                (x_min, x_max), (y_min, y_max) = self.imview.getView().viewRange()

                x_min = max(int(np.floor(x_min)), 0)
                x_max = min(int(np.ceil(x_max)), im.shape[1])
                y_min = max(int(np.floor(y_min)), 0)
                y_max = min(int(np.ceil(y_max)), im.shape[0])

                roi = i[y_min:y_max, x_min:x_max]
                if roi.size == 0:
                    print(f"{ind} empty ROI")
                    continue
            else:
                roi = im

            corners_found, corners = cv.findChessboardCornersSB(
                roi, (num_rows, num_cols), chess_flags
            )

            if corners_found:
                corners = cv.cornerSubPix(
                    roi, corners, (11, 11), (-1, -1), criteria
                )
                corners = _opencv_image_points(corners, f'checkerboard corners at frame {ind}')
                corners += [x_min, y_min]
                self.image_corners.append(corners)
                self.board_corners.append(board_pts.copy())
                self.cal_inds.append(int(ind))
            else:
                print(f"{ind} no corners found")

        # Need enough successful detections before calibrating
        if len(self.image_corners) < 3:
            print(f"Only found checkerboards in {len(self.image_corners)} frame(s); need at least 3.")
            return False

        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(
            self.board_corners,
            self.image_corners,
            (self.w, self.h),
            None,
            None,
        )

        self.nmtx, self.roi = cv.getOptimalNewCameraMatrix(
            self.mtx, self.dist, (self.w, self.h), 1, (self.w, self.h)
        )

        self.update_im()
        self.show_markers()
        return True

    def get_charuco_calibration(
            self, inds,
            num_rows=9,
            num_cols=12,
            side_len=15.0,
            marker_len=11.0,
            dictionary_id=None,
            legacy_pattern=False,
            use_roi=True):
        """Detect a ChArUco board and calibrate one camera.

        num_rows and num_cols are the numbers of chessboard squares, not the
        number of inner checkerboard corners. The ChArUco corner IDs are kept
        so the stereo calibration can use only the corners detected in both
        camera views for each frame.
        """
        if not hasattr(cv, 'aruco'):
            self.parent.console_write("This OpenCV build has no cv.aruco module.", "ChArUco")
            return False

        if marker_len is None:
            marker_len = side_len * 0.7
        if dictionary_id is None:
            # dictionary_id = cv.aruco.DICT_4X4_50
            dictionary_id = cv.aruco.DICT_5X5_100

        self.cal_inds = []
        self.image_corners = []
        self.board_corners = []
        self.image_corner_ids = []

        dictionary = cv.aruco.getPredefinedDictionary(dictionary_id)
        board = cv.aruco.CharucoBoard(
            (num_cols, num_rows),
            float(side_len),
            float(marker_len),
            dictionary
        )
        if legacy_pattern and hasattr(board, 'setLegacyPattern'):
            board.setLegacyPattern(True)

        board_corners = board.getChessboardCorners().astype(np.float32)
        min_corners = 6

        use_new_api = hasattr(cv.aruco, 'CharucoDetector')
        if use_new_api:
            detector = cv.aruco.CharucoDetector(board)
        else:
            detector = None
            detector_params = cv.aruco.DetectorParameters()

        for ind in inds:
            self.cap.set(cv.CAP_PROP_POS_FRAMES, ind)
            im_found, im = self.cap.read()

            if not im_found or im is None:
                print(f"{ind} no im found")
                continue

            gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

            mn, mx = self.imhist.item.getLevels()
            i = np.interp(gray, [mn, mx], [0, 255]).astype("uint8")

            if use_roi:
                (x_min, x_max), (y_min, y_max) = self.imview.getView().viewRange()
                x_min = max(int(np.floor(x_min)), 0)
                x_max = min(int(np.ceil(x_max)), gray.shape[1])
                y_min = max(int(np.floor(y_min)), 0)
                y_max = min(int(np.ceil(y_max)), gray.shape[0])

                roi = i[y_min:y_max, x_min:x_max]
            else:
                x_min, y_min = 0, 0
                roi = gray

            if roi.size == 0:
                print(f"{ind} empty ROI")
                continue

            if use_new_api:
                charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(roi)
            else:
                marker_corners, marker_ids, _ = cv.aruco.detectMarkers(
                    roi, dictionary, parameters=detector_params
                )
                if marker_ids is None or len(marker_ids) == 0:
                    charuco_corners, charuco_ids = None, None
                else:
                    _, charuco_corners, charuco_ids = cv.aruco.interpolateCornersCharuco(
                        marker_corners, marker_ids, roi, board
                    )

            marker_count = 0 if marker_ids is None else len(marker_ids)
            corner_count = 0 if charuco_ids is None else len(charuco_ids)
            print(f"{ind}: ArUco markers={marker_count}, ChArUco corners={corner_count}")

            if charuco_ids is None or charuco_corners is None or len(charuco_ids) < min_corners:
                print(f"{ind} only {corner_count} ChArUco corners found")
                continue

            charuco_corners = _opencv_image_points(
                charuco_corners, f'ChArUco corners at frame {ind}'
            )
            charuco_ids = _opencv_corner_ids(
                charuco_ids, f'ChArUco IDs at frame {ind}'
            )
            if len(charuco_corners) != len(charuco_ids):
                raise ValueError(
                    f'ChArUco corner/ID count mismatch at frame {ind}: '
                    f'corners {charuco_corners.shape}, IDs {charuco_ids.shape}'
                )
            charuco_corners += [x_min, y_min]
            obj_pts = board_corners[charuco_ids.flatten()]

            self.image_corners.append(charuco_corners)
            self.board_corners.append(obj_pts)
            self.image_corner_ids.append(charuco_ids)
            self.cal_inds.append(int(ind))

        if len(self.image_corners) < 3:
            print(f"Only found ChArUco boards in {len(self.image_corners)} frame(s); need at least 3.")
            return False

        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(
            self.board_corners,
            self.image_corners,
            (self.w, self.h),
            None,
            None,
        )

        self.nmtx, self.roi = cv.getOptimalNewCameraMatrix(
            self.mtx, self.dist, (self.w, self.h), 1, (self.w, self.h)
        )

        self.update_im()
        self.show_markers()

        return True

    def get_circle_calibration(self, inds, num_rows=4, num_cols=11, side_len=10.0,
                               use_roi=True, grid_type='asymmetric_equal',
                               first_row_count=None, second_row_count=None):
        self.cal_inds = []
        self.image_corners = []
        self.board_corners = []
        self.image_corner_ids = []

        candidates = circle_grid_detection_candidates(
            num_cols,
            num_rows,
            side_len,
            grid_type=grid_type,
            first_row_count=first_row_count,
            second_row_count=second_row_count,
        )

        for ind in inds:
            self.cap.set(cv.CAP_PROP_POS_FRAMES, ind)
            im_found, im = self.cap.read()

            if not im_found or im is None:
                print(f"{ind} no im found")
                continue

            gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

            x_min, y_min = 0, 0
            if use_roi:
                mn, mx = self.imhist.item.getLevels()
                i = np.interp(gray, [mn, mx], [0, 255]).astype("uint8")

                (x_min, x_max), (y_min, y_max) = self.imview.getView().viewRange()
                x_min = max(int(np.floor(x_min)), 0)
                x_max = min(int(np.ceil(x_max)), gray.shape[1])
                y_min = max(int(np.floor(y_min)), 0)
                y_max = min(int(np.ceil(y_max)), gray.shape[0])

                roi = i[y_min:y_max, x_min:x_max]
                if roi.size == 0:
                    print(f"{ind} empty ROI")
                    continue
            else:
                roi = gray

            centers_found, centers, candidate = find_circle_grid_with_candidates(roi, candidates)

            if centers_found:
                centers = _opencv_image_points(centers, f'circle-grid centers at frame {ind}')
                centers += [x_min, y_min]
                self.image_corners.append(centers)
                self.board_corners.append(candidate['object_points'].copy())
                self.cal_inds.append(int(ind))
                print(f"{ind} circle grid found: {candidate['name']}")
            else:
                print(f"{ind} no circle grid found")

        if len(self.image_corners) < 3:
            print(f"Only found circle grids in {len(self.image_corners)} frame(s); need at least 3.")
            return False

        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(
            self.board_corners,
            self.image_corners,
            (self.w, self.h),
            None,
            None
        )

        self.nmtx, self.roi = cv.getOptimalNewCameraMatrix(
            self.mtx, self.dist, (self.w, self.h), 1, (self.w, self.h)
        )

        self.update_im()
        self.show_markers()
        return True


##########
# Console
##########
class Console(pyqtgraph.console.ConsoleWidget):

    def __init__(self, st_window, namespace, text):
        super().__init__(namespace=namespace, text=text)
        self.show()
        self.st_window = st_window

    def write(self, text):
        """Append text to the console output widget."""
        if hasattr(self.output, "appendPlainText"):
            self.output.appendPlainText(text)
        else:
            self.output.append(text)

    def enterEvent(self, event):
        self.input.setFocus()
        self.st_window.releaseKeyboard()

    def leaveEvent(self, event):
        self.st_window.setFocus()
        self.st_window.grabKeyboard()


############################
### Board calibration dialog
############################

def _circle_grid_points_symmetric(cols, rows, spacing):
    pts = np.zeros((cols * rows, 3), np.float32)
    k = 0
    for r in range(rows):
        for c in range(cols):
            pts[k, 0] = c * spacing
            pts[k, 1] = r * spacing
            k += 1
    return pts


def _circle_grid_points_asymmetric_equal(cols, rows, spacing):
    pts = np.zeros((cols * rows, 3), np.float32)
    k = 0
    for r in range(rows):
        for c in range(cols):
            pts[k, 0] = (2 * c + (r % 2)) * spacing
            pts[k, 1] = r * spacing
            k += 1
    return pts


def _circle_grid_points_alternating(first_row_count, second_row_count, rows, spacing):
    row_counts = [
        first_row_count if r % 2 == 0 else second_row_count
        for r in range(rows)
    ]

    pts = np.zeros((sum(row_counts), 3), np.float32)
    short_count = min(first_row_count, second_row_count)
    k = 0
    for r, count in enumerate(row_counts):
        # Shorter rows are centered in the gaps of the longer rows.
        # For a 5/6 board this gives:
        #   5-center row: x = 1, 3, 5, 7, 9
        #   6-center row: x = 0, 2, 4, 6, 8, 10
        offset = 1 if count == short_count and first_row_count != second_row_count else 0
        for c in range(count):
            pts[k, 0] = (2 * c + offset) * spacing
            pts[k, 1] = r * spacing
            k += 1
    return pts


def circle_grid_detection_candidates(cols, rows, spacing, grid_type='asymmetric_equal',
                                     first_row_count=None, second_row_count=None):
    """Return possible OpenCV circle-grid interpretations.

    Each candidate has a pattern size for findCirclesGrid, flags, object
    points with matching length, and a descriptive name. The alternating-row
    case tries the common row-pair convention first, because some printed
    asymmetric grids are described as alternating short and long rows even though
    OpenCV detects them as a logical grid.
    """
    cols = max(1, int(cols))
    rows = max(1, int(rows))
    spacing = float(spacing)
    first_row_count = cols if first_row_count is None else max(1, int(first_row_count))
    second_row_count = cols if second_row_count is None else max(1, int(second_row_count))
    candidates = []

    if grid_type == 'symmetric':
        obj = _circle_grid_points_symmetric(cols, rows, spacing)
        candidates.append({
            'name': 'symmetric',
            'pattern_size': (cols, rows),
            'flags': cv.CALIB_CB_SYMMETRIC_GRID,
            'object_points': obj,
        })

    elif grid_type in ('asymmetric', 'asymmetric_alternating'):
        # Candidate 1: treat each short+long pair of physical rows as one
        # OpenCV logical asymmetric row. This matches boards visually described
        # as 5/6 alternating rows, where a row pair contains 11 centers.
        if rows % 2 == 0:
            obj = _circle_grid_points_alternating(
                first_row_count, second_row_count, rows, spacing
            )
            candidates.append({
                'name': f'alternating rows {first_row_count}/{second_row_count} row pairs',
                'pattern_size': (first_row_count + second_row_count, rows // 2),
                'flags': cv.CALIB_CB_ASYMMETRIC_GRID,
                'object_points': obj,
            })

        # Candidate 2: OpenCV's equal-row asymmetric interpretation, using the
        # long-row count as the pattern width. Some printed targets are labeled
        # this way even if the visual bounding columns alternate.
        max_row_count = max(first_row_count, second_row_count)
        obj = _circle_grid_points_asymmetric_equal(max_row_count, rows, spacing)
        candidates.append({
            'name': f'asymmetric equal rows, {max_row_count} × {rows}',
            'pattern_size': (max_row_count, rows),
            'flags': cv.CALIB_CB_ASYMMETRIC_GRID,
            'object_points': obj,
        })

        # Candidate 3: same idea, but using the total short+long count as the
        # OpenCV pattern width. This is useful for testing ambiguous printed
        # descriptions such as "11×4" with 5/6 visible row counts.
        total_pair_count = first_row_count + second_row_count
        obj = _circle_grid_points_asymmetric_equal(total_pair_count, rows, spacing)
        candidates.append({
            'name': f'asymmetric logical columns, {total_pair_count} × {rows}',
            'pattern_size': (total_pair_count, rows),
            'flags': cv.CALIB_CB_ASYMMETRIC_GRID,
            'object_points': obj,
        })

    else:
        obj = _circle_grid_points_asymmetric_equal(cols, rows, spacing)
        candidates.append({
            'name': f'asymmetric equal rows, {cols} × {rows}',
            'pattern_size': (cols, rows),
            'flags': cv.CALIB_CB_ASYMMETRIC_GRID,
            'object_points': obj,
        })

    # For circle grids, the clustering algorithm can rescue some strongly
    # perspective-distorted views, but it can also be more sensitive to clutter.
    # Try the plain version first, then a clustered version of each candidate.
    clustered = []
    for cand in candidates:
        clustered.append({**cand, 'name': cand['name'] + ' + clustering',
                          'flags': cand['flags'] | cv.CALIB_CB_CLUSTERING})
    return candidates + clustered


def find_circle_grid_with_candidates(image, candidates):
    """Try circle-grid candidates and return the first successful detection."""
    for cand in candidates:
        found, centers = cv.findCirclesGrid(
            image,
            cand['pattern_size'],
            flags=cand['flags']
        )
        if found and centers is not None:
            if len(centers) == len(cand['object_points']):
                return True, centers, cand
    return False, None, None


class BoardPreview(QtWidgets.QWidget):
    """Small live preview of board geometry and orientation."""

    def __init__(self, squares_x=9, squares_y=12, board_type='charuco', parent=None):
        super().__init__(parent)
        self.squares_x = int(squares_x)
        self.squares_y = int(squares_y)
        self.board_type = board_type
        self.circle_grid_type = 'asymmetric'
        self.first_row_count = 5
        self.second_row_count = 6
        self.setMinimumSize(220, 180)
        self.setMaximumSize(300, 240)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)

    def set_squares(self, squares_x, squares_y):
        self.squares_x = max(1, int(squares_x))
        self.squares_y = max(1, int(squares_y))
        self.update()

    def set_board_type(self, board_type):
        self.board_type = board_type
        self.update()

    def set_circle_grid(self, grid_type=None, first_row_count=None, second_row_count=None):
        if grid_type is not None:
            self.circle_grid_type = grid_type
        if first_row_count is not None:
            self.first_row_count = max(1, int(first_row_count))
        if second_row_count is not None:
            self.second_row_count = max(1, int(second_row_count))
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, self.board_type == 'circle_grid')

        rect = self.rect().adjusted(12, 12, -12, -30)
        if rect.width() <= 0 or rect.height() <= 0:
            return

        if self.board_type == 'circle_grid':
            self._paint_circle_grid(painter, rect)
        else:
            self._paint_checkerboard(painter, rect)

        painter.setPen(QtGui.QColor(0, 0, 0))
        if self.board_type == 'circle_grid':
            if self.circle_grid_type in ('asymmetric', 'asymmetric_alternating'):
                label = (
                    f"odd {self.first_row_count} / even {self.second_row_count} columns "
                    f"× {self.squares_y} rows"
                )
            else:
                label = f"{self.squares_x} centers × {self.squares_y} rows"
        else:
            label = f"{self.squares_x} columns × {self.squares_y} rows"
        painter.drawText(
            QtCore.QRect(0, self.height() - 24, self.width(), 20),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            label
        )

    def _fit_rect(self, rect, aspect):
        rect_aspect = rect.width() / rect.height()
        if aspect >= rect_aspect:
            w = rect.width()
            h = int(w / aspect)
        else:
            h = rect.height()
            w = int(h * aspect)
        x0 = rect.x() + (rect.width() - w) // 2
        y0 = rect.y() + (rect.height() - h) // 2
        return x0, y0, w, h

    def _paint_checkerboard(self, painter, rect):
        board_aspect = self.squares_x / self.squares_y
        x0, y0, board_w, board_h = self._fit_rect(rect, board_aspect)

        square_w = board_w / self.squares_x
        square_h = board_h / self.squares_y

        for y in range(self.squares_y):
            for x in range(self.squares_x):
                color = QtGui.QColor(30, 30, 30) if (x + y) % 2 == 0 else QtGui.QColor(245, 245, 245)
                painter.fillRect(
                    QtCore.QRectF(x0 + x * square_w, y0 + y * square_h, square_w, square_h),
                    color
                )

        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 1))
        painter.drawRect(QtCore.QRectF(x0, y0, board_w, board_h))

    def _paint_circle_grid(self, painter, rect):
        if self.circle_grid_type == 'symmetric':
            points = [(c, r) for r in range(max(1, self.squares_y))
                      for c in range(max(1, self.squares_x))]
        elif self.circle_grid_type in ('asymmetric', 'asymmetric_alternating'):
            points = self._alternating_circle_points()
        else:
            # Preview OpenCV's equal-row asymmetric circle grid. Adjacent rows
            # are offset by half of the horizontal center spacing.
            cols = max(1, self.squares_x)
            rows = max(1, self.squares_y)
            points = [(2 * c + (r % 2), r) for r in range(rows) for c in range(cols)]

        if not points:
            return

        xs = [pt[0] for pt in points]
        ys = [pt[1] for pt in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        model_w = max(1.0, max_x - min_x)
        model_h = max(1.0, max_y - min_y)
        aspect = model_w / model_h if model_h else 1.0
        x0, y0, board_w, board_h = self._fit_rect(rect, aspect)

        sx = board_w / model_w if model_w else board_w
        sy = board_h / model_h if model_h else board_h
        scale = min(sx, sy)
        radius = max(3.0, 0.28 * scale)

        painter.fillRect(QtCore.QRectF(x0, y0, board_w, board_h), QtGui.QColor(245, 245, 245))
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 1))
        painter.drawRect(QtCore.QRectF(x0, y0, board_w, board_h))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)

        for x, y in points:
            cx = x0 + (x - min_x) * sx
            cy = y0 + (y - min_y) * sy
            painter.drawEllipse(QtCore.QPointF(cx, cy), radius, radius)

    def _alternating_circle_points(self):
        rows = max(1, self.squares_y)
        first_n = max(1, self.first_row_count)
        second_n = max(1, self.second_row_count)
        short_n = min(first_n, second_n)
        points = []
        for row in range(rows):
            count = first_n if row % 2 == 0 else second_n
            # Center shorter rows in the gaps of the longer rows.
            # This avoids putting the extra circle only off one side.
            offset = 1 if count == short_n and first_n != second_n else 0
            for col in range(count):
                points.append((2 * col + offset, row))
        return points


class CharucoDetectionPreview(QtWidgets.QWidget):
    """Show board-test detections for the current stereo frame pair."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QtWidgets.QLabel('Test preview')
        title.setStyleSheet('font-weight: bold;')
        layout.addWidget(title)

        self.summary = QtWidgets.QLabel('Click Test to detect the board in the current frame pair.')
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        image_row = QtWidgets.QHBoxLayout()
        layout.addLayout(image_row)

        self.image_labels = []
        for label_text in ('Camera 0', 'Camera 1'):
            box = QtWidgets.QVBoxLayout()
            label = QtWidgets.QLabel(label_text)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            image = QtWidgets.QLabel()
            image.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            image.setMinimumSize(260, 200)
            image.setFrameShape(QtWidgets.QFrame.Shape.Box)
            image.setScaledContents(False)
            box.addWidget(label)
            box.addWidget(image)
            image_row.addLayout(box)
            self.image_labels.append(image)

        self.details = QtWidgets.QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMinimumHeight(130)
        layout.addWidget(self.details)

    def clear(self):
        self.summary.setText('Click Test to detect the board in the current frame pair.')
        self.details.clear()
        for image in self.image_labels:
            image.clear()

    def set_results(self, results, shared_count=None, board_type='charuco'):
        lines = []
        for result in results:
            if not result.get('ok'):
                lines.append(f"Camera {result['camera']}: {result.get('error', 'no image')}")
                continue

            if board_type == 'checkerboard':
                if result.get('found'):
                    lines.append(
                        f"Camera {result['camera']}: "
                        f"checkerboard found, {result['corner_count']}/{result['expected_corners']} corners"
                    )
                else:
                    lines.append(
                        f"Camera {result['camera']}: "
                        f"checkerboard not found, 0/{result['expected_corners']} corners"
                    )
            elif board_type == 'circle_grid':
                if result.get('found'):
                    mode = result.get('circle_mode')
                    mode_text = f" ({mode})" if mode else ""
                    lines.append(
                        f"Camera {result['camera']}: "
                        f"circle grid found, {result['corner_count']}/{result['expected_corners']} centers"
                        f"{mode_text}"
                    )
                else:
                    lines.append(
                        f"Camera {result['camera']}: "
                        f"circle grid not found, 0/{result['expected_corners']} centers"
                    )
            else:
                lines.append(
                    f"Camera {result['camera']}: "
                    f"{result['marker_count']} ArUco markers, "
                    f"{result['corner_count']} ChArUco corners"
                )

        if shared_count is not None:
            lines.append(f"Shared ChArUco corners: {shared_count}")

        self.summary.setText('\n'.join(lines))
        self.details.setPlainText('\n'.join(lines))

        for result, image_label in zip(results, self.image_labels):
            pixmap = result.get('pixmap')
            if pixmap is None:
                image_label.clear()
                continue
            image_label.setPixmap(
                pixmap.scaled(
                    image_label.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation
                )
            )


class BoardCalibrationDialog(QtWidgets.QDialog):
    """Collect board-calibration settings and test ChArUco detections."""

    def __init__(self, st_win, parent=None):
        super().__init__(parent)
        self.st_win = st_win
        self.qsettings = QtCore.QSettings('TheobaldLab', 'Jink3D')
        self.setWindowTitle('Camera geometry')
        self.setModal(True)
        self.setMinimumWidth(1050)

        outer = QtWidgets.QHBoxLayout(self)

        left = QtWidgets.QVBoxLayout()
        outer.addLayout(left, 0)

        form = QtWidgets.QFormLayout()
        left.addLayout(form)

        self.board_type = QtWidgets.QComboBox()
        self.board_type.addItem('ChArUco board', 'charuco')
        self.board_type.addItem('Checkerboard', 'checkerboard')
        self.board_type.addItem('Circle grid', 'circle_grid')
        form.addRow('Board type:', self.board_type)
        self._set_combo_data(self.board_type, self.qsettings.value('board_calibration/board_type', 'charuco'))

        self.circle_grid_type = QtWidgets.QComboBox()
        self.circle_grid_type.addItem('Symmetric circle grid', 'symmetric')
        self.circle_grid_type.addItem('Asymmetric circle grid', 'asymmetric')
        saved_grid_type = self.qsettings.value('circle_grid/grid_type', 'asymmetric')
        if saved_grid_type in ('asymmetric_equal', 'asymmetric_alternating'):
            saved_grid_type = 'asymmetric'
        self._set_combo_data(self.circle_grid_type, saved_grid_type)
        form.addRow('Circle grid type:', self.circle_grid_type)

        self.first_row_count = QtWidgets.QSpinBox()
        self.first_row_count.setRange(1, 64)
        self.first_row_count.setMaximumWidth(58)
        self.first_row_count.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.first_row_count.setValue(
            self._setting_int('circle_grid/odd_col_count', self._setting_int('circle_grid/first_row_count', 5)))

        self.second_row_count = QtWidgets.QSpinBox()
        self.second_row_count.setRange(1, 64)
        self.second_row_count.setMaximumWidth(58)
        self.second_row_count.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.second_row_count.setValue(
            self._setting_int('circle_grid/even_col_count', self._setting_int('circle_grid/second_row_count', 6)))

        self.squares_x = QtWidgets.QSpinBox()
        self.squares_x.setRange(2, 64)
        self.squares_x.setMaximumWidth(58)
        self.squares_x.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.squares_x.setValue(self._setting_int('charuco/squares_x', getattr(st_win, 'charuco_squares_x', 9)))

        self.squares_y = QtWidgets.QSpinBox()
        self.squares_y.setRange(2, 64)
        self.squares_y.setMaximumWidth(58)
        self.squares_y.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.squares_y.setValue(self._setting_int('charuco/squares_y', getattr(st_win, 'charuco_squares_y', 12)))

        self.board_preview = BoardPreview(self.squares_x.value(), self.squares_y.value())

        # Place columns above the preview and rows beside it so the controls
        # match the board orientation visually. The spin boxes are intentionally
        # narrow because realistic square counts are one or two digits.
        board_box = QtWidgets.QGridLayout()
        board_box.setHorizontalSpacing(8)
        board_box.setVerticalSpacing(4)

        self.col_label = QtWidgets.QLabel('columns')
        self.col_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.row_label = QtWidgets.QLabel('rows')
        self.row_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        col_box = QtWidgets.QVBoxLayout()
        col_box.setContentsMargins(0, 0, 0, 0)
        col_box.addWidget(self.col_label)
        col_box.addWidget(self.squares_x, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        normal_col_widget = QtWidgets.QWidget()
        normal_col_widget.setLayout(col_box)

        odd_box = QtWidgets.QVBoxLayout()
        odd_box.setContentsMargins(0, 0, 0, 0)
        self.odd_col_label = QtWidgets.QLabel('odd columns')
        self.odd_col_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        odd_box.addWidget(self.odd_col_label)
        odd_box.addWidget(self.first_row_count, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        even_box = QtWidgets.QVBoxLayout()
        even_box.setContentsMargins(0, 0, 0, 0)
        self.even_col_label = QtWidgets.QLabel('even columns')
        self.even_col_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        even_box.addWidget(self.even_col_label)
        even_box.addWidget(self.second_row_count, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        asym_col_box = QtWidgets.QHBoxLayout()
        asym_col_box.setContentsMargins(0, 0, 0, 0)
        asym_col_box.setSpacing(16)
        asym_col_box.addLayout(odd_box)
        asym_col_box.addLayout(even_box)
        asym_col_widget = QtWidgets.QWidget()
        asym_col_widget.setLayout(asym_col_box)

        self.column_stack = QtWidgets.QStackedWidget()
        self.column_stack.addWidget(normal_col_widget)
        self.column_stack.addWidget(asym_col_widget)

        row_box = QtWidgets.QHBoxLayout()
        row_box.setContentsMargins(0, 0, 0, 0)
        row_box.addWidget(self.row_label)
        row_box.addWidget(self.squares_y)

        board_box.addWidget(self.column_stack, 0, 1, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        board_box.addLayout(row_box, 1, 0, alignment=QtCore.Qt.AlignmentFlag.AlignVCenter)
        board_box.addWidget(self.board_preview, 1, 1)

        form.addRow('Squares:', board_box)

        self.square_len = QtWidgets.QDoubleSpinBox()
        self.square_len.setRange(0.001, 10000.0)
        self.square_len.setDecimals(3)
        self.square_len.setSingleStep(1.0)
        self.square_len.setMaximumWidth(86)
        self.square_len.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.square_len.setValue(self._setting_float('charuco/square_len', getattr(st_win, 'charuco_square_len', 15.0)))

        self.marker_len = QtWidgets.QDoubleSpinBox()
        self.marker_len.setRange(0.001, 10000.0)
        self.marker_len.setDecimals(3)
        self.marker_len.setSingleStep(1.0)
        self.marker_len.setMaximumWidth(86)
        self.marker_len.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.marker_len.setValue(self._setting_float('charuco/marker_len', getattr(st_win, 'charuco_marker_len', 11.0)))

        size_box = QtWidgets.QHBoxLayout()
        size_box.setContentsMargins(0, 0, 0, 0)
        size_box.setSpacing(10)
        self.size_left_label = QtWidgets.QLabel('square')
        self.size_right_label = QtWidgets.QLabel('marker')
        size_box.addWidget(self.size_left_label)
        size_box.addWidget(self.square_len)
        size_box.addSpacing(12)
        size_box.addWidget(self.size_right_label)
        size_box.addWidget(self.marker_len)
        size_box.addStretch()
        form.addRow('Sizes:', size_box)

        self.dictionary = QtWidgets.QComboBox()
        self._add_dictionaries()
        default_dict = getattr(st_win, 'charuco_dictionary', None)
        saved_dict = self._setting_int('charuco/dictionary_id', default_dict) if default_dict is not None else None
        self._set_current_dictionary(saved_dict)
        form.addRow('Dictionary:', self.dictionary)

        self.frame_mode = QtWidgets.QComboBox()
        self.frame_mode.addItem('Marked frames; if none, evenly spaced', 'marked_or_even')
        self.frame_mode.addItem('Evenly spaced frames', 'even')
        self.frame_mode.addItem('Current frame only', 'current')
        self.frame_mode.addItem('Frame range', 'range')
        form.addRow('Frames:', self.frame_mode)
        self._set_combo_data(self.frame_mode, self.qsettings.value('charuco/frame_mode', 'marked_or_even'))

        self.num_frames = QtWidgets.QSpinBox()
        self.num_frames.setRange(1, 500)
        self.num_frames.setValue(self._setting_int('charuco/num_frames', 20))
        form.addRow('Number of frames:', self.num_frames)

        range_box = QtWidgets.QHBoxLayout()
        self.start_frame = QtWidgets.QSpinBox()
        self.stop_frame = QtWidgets.QSpinBox()
        self.frame_step = QtWidgets.QSpinBox()
        max_frame = max(0, int(getattr(st_win, 'num_frames', 1)) - 1)
        self.start_frame.setRange(0, max_frame)
        self.stop_frame.setRange(0, max_frame)
        self.start_frame.setValue(min(max_frame, self._setting_int('charuco/start_frame', 0)))
        self.stop_frame.setValue(min(max_frame, self._setting_int('charuco/stop_frame', max_frame)))
        self.frame_step.setRange(1, max(1, max_frame))
        default_step = max(1, max_frame // 19) if max_frame else 1
        self.frame_step.setValue(min(max(1, max_frame), self._setting_int('charuco/frame_step', default_step)))
        range_box.addWidget(QtWidgets.QLabel('start'))
        range_box.addWidget(self.start_frame)
        range_box.addWidget(QtWidgets.QLabel('stop'))
        range_box.addWidget(self.stop_frame)
        range_box.addWidget(QtWidgets.QLabel('step'))
        range_box.addWidget(self.frame_step)
        form.addRow('Frame range:', range_box)

        self.region = QtWidgets.QComboBox()
        self.region.addItem('Current visible region', True)
        self.region.addItem('Full frame', False)
        saved_use_roi = self._setting_bool('charuco/use_roi', True)
        self._set_combo_data(self.region, saved_use_roi)
        form.addRow('Detection region:', self.region)

        self.legacy_pattern = QtWidgets.QCheckBox('Use legacy ChArUco pattern')
        self.legacy_pattern.setChecked(self._setting_bool('charuco/legacy_pattern', False))
        form.addRow('', self.legacy_pattern)

        self.note = QtWidgets.QLabel('Choose a camera-geometry target, then Test or OK.')
        self.note.setWordWrap(True)
        left.addWidget(self.note)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.test_button = buttons.addButton('Test', QtWidgets.QDialogButtonBox.ActionRole)
        self.test_button.clicked.connect(self._test_current_frame)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        left.addWidget(buttons)

        self.detection_preview = CharucoDetectionPreview()
        outer.addWidget(self.detection_preview, 1)

        self.board_type.currentIndexChanged.connect(self._board_type_changed)
        self.frame_mode.currentIndexChanged.connect(self._frame_mode_changed)
        self.squares_x.valueChanged.connect(self._update_board_preview)
        self.squares_y.valueChanged.connect(self._update_board_preview)
        self.circle_grid_type.currentIndexChanged.connect(self._circle_grid_type_changed)
        self.first_row_count.valueChanged.connect(self._first_row_count_changed)
        self.second_row_count.valueChanged.connect(self._second_row_count_changed)
        self._update_board_preview()
        self._board_type_changed()
        self._frame_mode_changed()

    def _setting_int(self, key, default):
        value = self.qsettings.value(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def _setting_float(self, key, default):
        value = self.qsettings.value(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _setting_bool(self, key, default):
        value = self.qsettings.value(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('1', 'true', 'yes')
        return bool(value)

    def _set_combo_data(self, combo, value):
        for index in range(combo.count()):
            if combo.itemData(index) == value or str(combo.itemData(index)) == str(value):
                combo.setCurrentIndex(index)
                return

    def _update_board_preview(self):
        self.board_preview.set_squares(self.squares_x.value(), self.squares_y.value())
        if hasattr(self, 'circle_grid_type'):
            self.board_preview.set_circle_grid(
                grid_type=self.circle_grid_type.currentData(),
                first_row_count=self.first_row_count.value(),
                second_row_count=self.second_row_count.value()
            )

    def _constrain_alternating_row_counts(self, changed):
        # Asymmetric circle grids are defined as neighboring rows whose column counts differ
        # by at most one. When one box is edited, gently pull the other back
        # into the valid range instead of allowing an impossible layout.
        first = self.first_row_count.value()
        second = self.second_row_count.value()
        if abs(first - second) <= 1:
            self._update_board_preview()
            return

        if changed == 'first':
            target = first + 1 if second > first else first - 1
            self.second_row_count.blockSignals(True)
            self.second_row_count.setValue(max(1, target))
            self.second_row_count.blockSignals(False)
        else:
            target = second + 1 if first > second else second - 1
            self.first_row_count.blockSignals(True)
            self.first_row_count.setValue(max(1, target))
            self.first_row_count.blockSignals(False)
        self._update_board_preview()

    def _first_row_count_changed(self):
        self._constrain_alternating_row_counts('first')

    def _second_row_count_changed(self):
        self._constrain_alternating_row_counts('second')

    def _circle_grid_type_changed(self):
        self._board_type_changed()
        self._update_board_preview()

    def _add_dictionaries(self):
        if not hasattr(cv, 'aruco'):
            self.dictionary.addItem('cv.aruco unavailable', None)
            return

        names = [
            'DICT_4X4_50', 'DICT_4X4_100', 'DICT_4X4_250', 'DICT_4X4_1000',
            'DICT_5X5_50', 'DICT_5X5_100', 'DICT_5X5_250', 'DICT_5X5_1000',
            'DICT_6X6_50', 'DICT_6X6_100', 'DICT_6X6_250', 'DICT_6X6_1000',
            'DICT_7X7_50', 'DICT_7X7_100', 'DICT_7X7_250', 'DICT_7X7_1000',
        ]
        for name in names:
            if hasattr(cv.aruco, name):
                self.dictionary.addItem(name, getattr(cv.aruco, name))

    def _set_current_dictionary(self, dict_id):
        if dict_id is None:
            return
        for index in range(self.dictionary.count()):
            if self.dictionary.itemData(index) == dict_id:
                self.dictionary.setCurrentIndex(index)
                return

    def _board_type_changed(self):
        board_type = self.board_type.currentData()
        is_charuco = board_type == 'charuco'
        is_checkerboard = board_type == 'checkerboard'
        is_circle_grid = board_type == 'circle_grid'
        is_supported = is_charuco or is_checkerboard or is_circle_grid
        circle_kind = self.circle_grid_type.currentData() if hasattr(self, 'circle_grid_type') else 'asymmetric'
        is_asymmetric_circle = is_circle_grid and circle_kind == 'asymmetric'
        self.board_preview.set_board_type(board_type)
        self._update_board_preview()

        # Column and row controls, preview, size and spacing, and detection region are
        # used by all implemented board types. For checkerboards these are
        # board squares; the calibration call subtracts one to get inner
        # intersections. For asymmetric circle grids, two compact column-count
        # boxes replace the single column-count box above the preview.
        for widget in [self.squares_x, self.squares_y, self.board_preview,
                       self.square_len, self.region]:
            widget.setEnabled(is_supported)

        # These fields only apply to ChArUco boards.
        for widget in [self.marker_len, self.dictionary, self.legacy_pattern]:
            widget.setEnabled(is_charuco)
        self.size_right_label.setVisible(is_charuco)
        self.marker_len.setVisible(is_charuco)

        # Circle-grid-only controls.
        self.circle_grid_type.setEnabled(is_circle_grid)
        self.circle_grid_type.setVisible(is_circle_grid)
        self.first_row_count.setEnabled(is_asymmetric_circle)
        self.second_row_count.setEnabled(is_asymmetric_circle)
        self.column_stack.setCurrentIndex(1 if is_asymmetric_circle else 0)

        self.col_label.setText('columns')
        self.row_label.setText('rows')
        self.size_left_label.setText('center spacing' if is_circle_grid else 'square')

        button_box = self.findChild(QtWidgets.QDialogButtonBox)
        if button_box is not None:
            button_box.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(is_supported)
        if hasattr(self, 'test_button'):
            self.test_button.setEnabled(is_supported)

        if is_charuco:
            self.note.setText(
                'ChArUco: columns and rows are board squares. Marker size and dictionary must match the printed board.'
            )
        elif is_checkerboard:
            self.note.setText(
                'Checkerboard: columns and rows are board squares in the preview; calibration uses one fewer inner corner in each direction.'
            )
        elif is_circle_grid:
            if circle_kind == 'symmetric':
                self.note.setText(
                    'Symmetric circle grid: columns and rows are circle centers. Size is center-to-center spacing.')
            else:
                self.note.setText(
                    'Asymmetric circle grid: odd and even column counts repeat down the board and must differ by at most one. Rows are total circle rows. Size is nearest-neighbor center spacing. Test this mode before calibration because OpenCV circle-grid conventions vary.'
                )
        else:
            self.note.setText('This board type is a placeholder for the later unified calibration dialog.')

    def _frame_mode_changed(self):
        mode = self.frame_mode.currentData()
        self.num_frames.setEnabled(mode in ('marked_or_even', 'even'))
        for widget in (self.start_frame, self.stop_frame, self.frame_step):
            widget.setEnabled(mode == 'range')

    def settings(self):
        return {
            'board_type': self.board_type.currentData(),
            'squares_x': int(self.squares_x.value()),
            'squares_y': int(self.squares_y.value()),
            'square_len': float(self.square_len.value()),
            'marker_len': float(self.marker_len.value()),
            'circle_grid_type': self.circle_grid_type.currentData(),
            'odd_col_count': int(self.first_row_count.value()),
            'even_col_count': int(self.second_row_count.value()),
            'first_row_count': int(self.first_row_count.value()),
            'second_row_count': int(self.second_row_count.value()),
            'dictionary_id': self.dictionary.currentData(),
            'frame_mode': self.frame_mode.currentData(),
            'num_frames': int(self.num_frames.value()),
            'start_frame': int(self.start_frame.value()),
            'stop_frame': int(self.stop_frame.value()),
            'frame_step': int(self.frame_step.value()),
            'use_roi': bool(self.region.currentData()),
            'legacy_pattern': self.legacy_pattern.isChecked(),
        }

    def _save_settings(self):
        settings = self.settings()
        self.qsettings.setValue('board_calibration/board_type', settings['board_type'])
        self.qsettings.setValue('charuco/squares_x', settings['squares_x'])
        self.qsettings.setValue('charuco/squares_y', settings['squares_y'])
        self.qsettings.setValue('charuco/square_len', settings['square_len'])
        self.qsettings.setValue('charuco/marker_len', settings['marker_len'])
        self.qsettings.setValue('charuco/dictionary_id', settings['dictionary_id'])
        self.qsettings.setValue('charuco/frame_mode', settings['frame_mode'])
        self.qsettings.setValue('charuco/num_frames', settings['num_frames'])
        self.qsettings.setValue('charuco/start_frame', settings['start_frame'])
        self.qsettings.setValue('charuco/stop_frame', settings['stop_frame'])
        self.qsettings.setValue('charuco/frame_step', settings['frame_step'])
        self.qsettings.setValue('charuco/use_roi', settings['use_roi'])
        self.qsettings.setValue('charuco/legacy_pattern', settings['legacy_pattern'])
        self.qsettings.setValue('circle_grid/grid_type', settings['circle_grid_type'])
        self.qsettings.setValue('circle_grid/odd_col_count', settings['odd_col_count'])
        self.qsettings.setValue('circle_grid/even_col_count', settings['even_col_count'])
        self.qsettings.setValue('circle_grid/first_row_count', settings['first_row_count'])
        self.qsettings.setValue('circle_grid/second_row_count', settings['second_row_count'])
        self.qsettings.sync()
        self._apply_to_window(settings)

    def _apply_to_window(self, settings):
        board_type = settings.get('board_type')
        self.st_win.cal_cols = settings['squares_x'] - 1
        self.st_win.cal_rows = settings['squares_y'] - 1
        self.st_win.cal_side = settings['square_len']

        if board_type == 'charuco':
            self.st_win.charuco_squares_x = settings['squares_x']
            self.st_win.charuco_squares_y = settings['squares_y']
            self.st_win.charuco_square_len = settings['square_len']
            self.st_win.charuco_marker_len = settings['marker_len']
            self.st_win.charuco_dictionary = settings['dictionary_id']

    def accept(self):
        self._save_settings()
        super().accept()

    def _test_current_frame(self):
        self._save_settings()
        settings = self.settings()
        board_type = settings['board_type']

        results = []
        id_sets = []
        frame_ind = int(self.st_win.frame_slider.value())
        for cam_ind, im in enumerate(self.st_win.ims):
            if board_type == 'charuco':
                result = self._detect_charuco_in_imframe(im, cam_ind, frame_ind, settings)
                ids = result.get('charuco_ids')
                id_sets.append(set(ids.flatten().tolist()) if ids is not None else set())
            elif board_type == 'checkerboard':
                result = self._detect_checkerboard_in_imframe(im, cam_ind, frame_ind, settings)
            elif board_type == 'circle_grid':
                result = self._detect_circle_grid_in_imframe(im, cam_ind, frame_ind, settings)
            else:
                self.detection_preview.summary.setText('This board type is not implemented yet.')
                return
            results.append(result)

        shared = None
        if board_type == 'charuco' and len(id_sets) == 2:
            shared = len(id_sets[0].intersection(id_sets[1]))

        self.detection_preview.set_results(results, shared, board_type=board_type)

    def _detect_charuco_in_imframe(self, im, cam_ind, frame_ind, settings):
        im.cap.set(cv.CAP_PROP_POS_FRAMES, frame_ind)
        ok, frame = im.cap.read()
        if not ok or frame is None:
            return {'camera': cam_ind, 'ok': False, 'error': f'could not read frame {frame_ind}'}

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        x_min, y_min = 0, 0
        if settings.get('use_roi', True):
            mn, mx = im.imhist.item.getLevels()
            detect_image = np.interp(gray, [mn, mx], [0, 255]).astype('uint8')
            (x0, x1), (y0, y1) = im.imview.getView().viewRange()
            x_min = max(int(np.floor(x0)), 0)
            x_max = min(int(np.ceil(x1)), gray.shape[1])
            y_min = max(int(np.floor(y0)), 0)
            y_max = min(int(np.ceil(y1)), gray.shape[0])
            roi = detect_image[y_min:y_max, x_min:x_max]
            if roi.size == 0:
                return {'camera': cam_ind, 'ok': False, 'error': f'empty ROI at frame {frame_ind}'}
        else:
            roi = gray

        dictionary = cv.aruco.getPredefinedDictionary(settings['dictionary_id'])
        board = cv.aruco.CharucoBoard(
            (int(settings['squares_x']), int(settings['squares_y'])),
            float(settings['square_len']),
            float(settings['marker_len']),
            dictionary
        )
        if settings.get('legacy_pattern', False) and hasattr(board, 'setLegacyPattern'):
            board.setLegacyPattern(True)

        if hasattr(cv.aruco, 'CharucoDetector'):
            detector = cv.aruco.CharucoDetector(board)
            charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(roi)
        else:
            detector_params = cv.aruco.DetectorParameters()
            marker_corners, marker_ids, _ = cv.aruco.detectMarkers(roi, dictionary, parameters=detector_params)
            if marker_ids is None or len(marker_ids) == 0:
                charuco_corners, charuco_ids = None, None
            else:
                _, charuco_corners, charuco_ids = cv.aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, roi, board
                )

        marker_count = 0 if marker_ids is None else len(marker_ids)
        corner_count = 0 if charuco_ids is None else len(charuco_ids)

        annotated = frame.copy()
        if marker_corners is not None and marker_ids is not None and len(marker_corners) > 0:
            marker_corners_full = [corners.copy().astype(np.float32) for corners in marker_corners]
            if x_min or y_min:
                for corners in marker_corners_full:
                    corners[:, :, 0] += x_min
                    corners[:, :, 1] += y_min
            cv.aruco.drawDetectedMarkers(annotated, marker_corners_full, marker_ids, borderColor=(0, 255, 255))

        charuco_corners_full = None
        if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 0:
            charuco_corners_full = _opencv_image_points(
                charuco_corners, f'ChArUco preview corners at frame {frame_ind}'
            )
            charuco_ids = _opencv_corner_ids(
                charuco_ids, f'ChArUco preview IDs at frame {frame_ind}'
            )
            if len(charuco_corners_full) != len(charuco_ids):
                raise ValueError(
                    f'ChArUco preview corner/ID count mismatch at frame {frame_ind}: '
                    f'corners {charuco_corners_full.shape}, IDs {charuco_ids.shape}'
                )
            charuco_corners_full[:, :, 0] += x_min
            charuco_corners_full[:, :, 1] += y_min
            cv.aruco.drawDetectedCornersCharuco(annotated, charuco_corners_full, charuco_ids, cornerColor=(0, 255, 255))

        pixmap = self._pixmap_from_bgr(annotated)
        return {
            'camera': cam_ind,
            'ok': True,
            'marker_count': marker_count,
            'corner_count': corner_count,
            'charuco_ids': charuco_ids,
            'pixmap': pixmap,
        }

    def _detect_checkerboard_in_imframe(self, im, cam_ind, frame_ind, settings):
        im.cap.set(cv.CAP_PROP_POS_FRAMES, frame_ind)
        ok, frame = im.cap.read()
        if not ok or frame is None:
            return {'camera': cam_ind, 'ok': False, 'error': f'could not read frame {frame_ind}'}

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        x_min, y_min = 0, 0
        if settings.get('use_roi', True):
            mn, mx = im.imhist.item.getLevels()
            detect_image = np.interp(gray, [mn, mx], [0, 255]).astype('uint8')
            (x0, x1), (y0, y1) = im.imview.getView().viewRange()
            x_min = max(int(np.floor(x0)), 0)
            x_max = min(int(np.ceil(x1)), gray.shape[1])
            y_min = max(int(np.floor(y0)), 0)
            y_max = min(int(np.ceil(y1)), gray.shape[0])
            roi = detect_image[y_min:y_max, x_min:x_max]
            if roi.size == 0:
                return {'camera': cam_ind, 'ok': False, 'error': f'empty ROI at frame {frame_ind}'}
        else:
            roi = gray

        # The dialog uses board squares for the visual preview. OpenCV's
        # checkerboard detector wants inner intersections.
        inner_x = max(1, int(settings['squares_x']) - 1)
        inner_y = max(1, int(settings['squares_y']) - 1)
        pattern_size = (inner_x, inner_y)
        chess_flags = (
                cv.CALIB_CB_ADAPTIVE_THRESH
                + cv.CALIB_CB_NORMALIZE_IMAGE
                + cv.CALIB_CB_FILTER_QUADS
        )
        corners_found, corners = cv.findChessboardCornersSB(roi, pattern_size, chess_flags)

        annotated = frame.copy()
        corner_count = 0
        if corners_found and corners is not None:
            corners_full = _opencv_image_points(
                corners, f'checkerboard preview corners at frame {frame_ind}'
            )
            corners_full[:, :, 0] += x_min
            corners_full[:, :, 1] += y_min
            corner_count = len(corners_full)
            for pt in np.asarray(corners_full).reshape(-1, 2):
                cv.circle(annotated, tuple(np.rint(pt).astype(int)), 4, (0, 255, 255), -1, cv.LINE_AA)

        pixmap = self._pixmap_from_bgr(annotated)
        return {
            'camera': cam_ind,
            'ok': True,
            'found': bool(corners_found),
            'corner_count': corner_count,
            'expected_corners': inner_x * inner_y,
            'pixmap': pixmap,
        }

    def _detect_circle_grid_in_imframe(self, im, cam_ind, frame_ind, settings):
        im.cap.set(cv.CAP_PROP_POS_FRAMES, frame_ind)
        ok, frame = im.cap.read()
        if not ok or frame is None:
            return {'camera': cam_ind, 'ok': False, 'error': f'could not read frame {frame_ind}'}

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        x_min, y_min = 0, 0
        if settings.get('use_roi', True):
            mn, mx = im.imhist.item.getLevels()
            detect_image = np.interp(gray, [mn, mx], [0, 255]).astype('uint8')
            (x0, x1), (y0, y1) = im.imview.getView().viewRange()
            x_min = max(int(np.floor(x0)), 0)
            x_max = min(int(np.ceil(x1)), gray.shape[1])
            y_min = max(int(np.floor(y0)), 0)
            y_max = min(int(np.ceil(y1)), gray.shape[0])
            roi = detect_image[y_min:y_max, x_min:x_max]
            if roi.size == 0:
                return {'camera': cam_ind, 'ok': False, 'error': f'empty ROI at frame {frame_ind}'}
        else:
            roi = gray

        cols = max(1, int(settings['squares_x']))
        rows = max(1, int(settings['squares_y']))
        candidates = circle_grid_detection_candidates(
            cols,
            rows,
            float(settings['square_len']),
            grid_type=settings.get('circle_grid_type', 'asymmetric'),
            first_row_count=settings.get('odd_col_count', settings.get('first_row_count', settings['squares_x'])),
            second_row_count=settings.get('even_col_count', settings.get('second_row_count', settings['squares_x'])),
        )
        centers_found, centers, candidate = find_circle_grid_with_candidates(roi, candidates)

        annotated = frame.copy()
        center_count = 0
        if centers_found and centers is not None:
            centers_full = _opencv_image_points(
                centers, f'circle-grid preview centers at frame {frame_ind}'
            )
            centers_full[:, :, 0] += x_min
            centers_full[:, :, 1] += y_min
            center_count = len(centers_full)
            for pt in np.asarray(centers_full).reshape(-1, 2):
                cv.circle(annotated, tuple(np.rint(pt).astype(int)), 4, (0, 255, 255), -1, cv.LINE_AA)

        pixmap = self._pixmap_from_bgr(annotated)
        expected = len(candidate['object_points']) if candidate is not None else len(candidates[0]['object_points'])
        return {
            'camera': cam_ind,
            'ok': True,
            'found': bool(centers_found),
            'corner_count': center_count,
            'expected_corners': expected,
            'circle_mode': None if candidate is None else candidate['name'],
            'pixmap': pixmap,
        }

    def _pixmap_from_bgr(self, image):
        rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w, ch = rgb.shape
        qimage = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888).copy()
        return QtGui.QPixmap.fromImage(qimage)


###############################
### Orientation calibration dialog
###############################

class OrientationCalibrationDialog(QtWidgets.QDialog):
    """Dialog for setting the reconstructed scene orientation."""

    def __init__(self, st_win, parent=None):
        super().__init__(parent)
        self.st_win = st_win
        self.setWindowTitle('Orientation of gravity')
        self.resize(640, 360)

        layout = QtWidgets.QVBoxLayout(self)

        intro = QtWidgets.QLabel(
            'These orientation methods come after camera geometry calibration. '
            'They rotate the reconstructed 3D coordinates so gravity points down.'
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.method = QtWidgets.QComboBox()
        self.method.addItem('Plumbline: marker 1 top, marker 2 bottom', 'plumbline')
        self.method.addItem('Thrown object: marker acceleration', 'projectile_center')
        form.addRow('Method:', self.method)

        self.projectile_widget = QtWidgets.QWidget()
        projectile_layout = QtWidgets.QFormLayout(self.projectile_widget)
        projectile_layout.setContentsMargins(0, 0, 0, 0)

        self.projectile_marker = QtWidgets.QSpinBox()
        self.projectile_marker.setRange(1, max(1, int(st_win.num_markers)))
        self.projectile_marker.setValue(1)
        self.projectile_marker.setMaximumWidth(60)
        projectile_layout.addRow('Thrown-object marker:', self.projectile_marker)

        self.projectile_fps = QtWidgets.QDoubleSpinBox()
        self.projectile_fps.setRange(0.001, 100000.0)
        self.projectile_fps.setDecimals(3)
        self.projectile_fps.setValue(self._default_fps())
        self.projectile_fps.setMaximumWidth(100)
        projectile_layout.addRow('Frame rate:', self.projectile_fps)

        form.addRow('', self.projectile_widget)

        self.note = QtWidgets.QLabel('')
        self.note.setWordWrap(True)
        layout.addWidget(self.note)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.method.currentIndexChanged.connect(self._method_changed)
        self._method_changed()

    def _default_fps(self):
        """Read matching camera configuration files, otherwise use 30 FPS."""
        fps_values = []
        for im in self.st_win.ims:
            video_fn = getattr(im, 'fn', '')
            if not video_fn:
                return 30.0

            config_fn = os.path.splitext(video_fn)[0] + '.cfg'
            parser = configparser.ConfigParser()
            try:
                with open(config_fn, encoding='utf-8') as config_file:
                    parser.read_file(config_file)
                fps = parser.getfloat('Record', 'fps')
            except (OSError, UnicodeError, configparser.Error, ValueError):
                return 30.0

            if not np.isfinite(fps) or fps <= 0:
                return 30.0
            fps_values.append(fps)

        if len(fps_values) != 2 or not np.isclose(fps_values[0], fps_values[1]):
            return 30.0
        return float(fps_values[0])

    def _method_changed(self):
        method = self.method.currentData()
        ok_button = self.findChild(QtWidgets.QDialogButtonBox).button(QtWidgets.QDialogButtonBox.Ok)

        self.projectile_widget.setVisible(method == 'projectile_center')

        if method == 'plumbline':
            self.note.setText(
                'Mark the top of the plumbline with marker 1 and the bottom with marker 2. '
                'Use at least one frame where both markers are present in both camera views.'
            )
            ok_button.setEnabled(True)

        elif method == 'projectile_center':
            self.note.setText(
                'Mark the center of a freely thrown object with the selected marker in both camera views. '
                'The program reconstructs the 3D trajectory, fits splines to x, y, and z, and uses the '
                'second derivative as the gravity vector. More marked frames generally give a better estimate.'
            )
            ok_button.setEnabled(True)

        else:
            ok_button.setEnabled(False)

    def settings(self):
        return {
            'method': self.method.currentData(),
            'projectile_marker_ind': int(self.projectile_marker.value()) - 1,
            'projectile_fps': float(self.projectile_fps.value()),
        }


####################
### stereography ###
####################
# sttt

class Stereography_window(QtWidgets.QMainWindow):  # QWidget
    '''Window with two camera views, top, camera placement and 3d render,
    bottom. This is the main window for pyqtgraph.

    '''

    def __init__(self, rot=False, cal_rows=8, cal_cols=11, cal_side=15):
        super(Stereography_window, self).__init__()
        self.window_title = 'Tracker 8'
        self.setWindowTitle(self.window_title)
        self.setMinimumWidth(900)
        self.setMinimumHeight(1000)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.cur = QtGui.QCursor

        self.num_cams = 2  # left and right
        self.num_markers = 9  # 1-9
        self.rot = True
        self.fns = ['', '']
        self.chk_calibration = ''
        self.pb_calibration = ''
        self.cal_rows = cal_rows
        self.cal_cols = cal_cols
        self.cal_side = cal_side
        self.charuco_marker_fraction = 0.7
        self.charuco_dictionary = cv.aruco.DICT_5X5_100 if hasattr(cv, 'aruco') else None
        self.charuco_squares_x = 9
        self.charuco_squares_y = 12
        self.charuco_square_len = 15.0
        self.charuco_marker_len = 11.0
        self.data_fn = ''
        self.get_data_fn = False
        self.npy_fn = ''
        self.get_npy_fn = False
        self.csv_fn = ''
        self.get_csv_fn = False
        # self.dir = ''
        self.curr_marker = 0

        ### menubar
        menubar = self.menuBar()

        #  file menu
        file_menu = menubar.addMenu('File')

        load_avi_action = QtWidgets.QAction('Open avis...', self)
        load_avi_action.triggered.connect(self.load_avis)
        load_avi_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_O)
        file_menu.addAction(load_avi_action)

        file_menu.addSeparator()

        load_data_action = QtWidgets.QAction('Load marker data...', self)
        load_data_action.triggered.connect(self.load_data)
        load_data_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_L)
        file_menu.addAction(load_data_action)

        file_menu.addSeparator()

        save_data_action = QtWidgets.QAction('Save marker data', self)
        save_data_action.triggered.connect(self.save_data)
        save_data_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_S)
        file_menu.addAction(save_data_action)

        save_data_as_action = QtWidgets.QAction('Save marker data as...', self)
        save_data_as_action.triggered.connect(self.save_data_as)
        save_data_as_action.setShortcut(QtCore.Qt.SHIFT + QtCore.Qt.CTRL + QtCore.Qt.Key_S)
        file_menu.addAction(save_data_as_action)

        file_menu.addSeparator()

        export_csv_action = QtWidgets.QAction('Export 3D data as CSV', self)
        export_csv_action.triggered.connect(self.export_csv)
        export_csv_as_action = QtWidgets.QAction('Export 3D data as CSV as...', self)
        export_csv_as_action.triggered.connect(self.export_csv_as)

        export_npy_action = QtWidgets.QAction('Export 3D data as npy...', self)
        export_npy_action.triggered.connect(self.export_npy)
        export_npy_action.setShortcut(QtCore.Qt.ALT + QtCore.Qt.Key_S)
        export_npy_as_action = QtWidgets.QAction('Export 3D data as npy as...', self)
        export_npy_as_action.triggered.connect(self.export_npy_as)

        file_menu.addActions([export_npy_action, export_npy_as_action, export_csv_action, export_csv_as_action])

        file_menu.addSeparator()

        quit_action = QtWidgets.QAction('Quit', self)
        quit_action.triggered.connect(self.close)
        # quit_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        file_menu.addAction(quit_action)

        # edit menu
        edit_menu = menubar.addMenu('Edit')

        undo_action = QtWidgets.QAction('Undo', self)
        undo_action.triggered.connect(self.undo)
        undo_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_Z)
        edit_menu.addAction(undo_action)

        # view menu
        view_menu = menubar.addMenu('View')

        reset_view_action = QtWidgets.QAction('Reset view', self)
        reset_view_action.triggered.connect(self.reset_views)
        reset_view_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_R)
        view_menu.addAction(reset_view_action)

        view_menu.addSeparator()

        quantile_lower_action = QtWidgets.QAction('Darker levels', self)
        quantile_lower_action.triggered.connect(self.adjust_lower_quantile)
        quantile_lower_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_D)

        quantile_middle_action = QtWidgets.QAction('Middle levels', self)
        quantile_middle_action.triggered.connect(self.adjust_middle_quantile)
        quantile_middle_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_M)

        quantile_upper_action = QtWidgets.QAction('Brighter levels', self)
        quantile_upper_action.triggered.connect(self.adjust_upper_quantile)
        quantile_upper_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_B)

        view_menu.addActions([quantile_lower_action, quantile_middle_action, quantile_upper_action])

        # navigate menu
        nav_menu = menubar.addMenu('Navigate')

        next_frame = QtWidgets.QAction('Next frame', self)
        next_frame.setShortcut(QtCore.Qt.Key_Right)
        next_frame.triggered.connect(self.next_frame)

        prev_frame = QtWidgets.QAction('Previous frame', self)
        prev_frame.setShortcut(QtCore.Qt.Key_Left)
        prev_frame.triggered.connect(self.prev_frame)

        next_frame_5 = QtWidgets.QAction('Forward 5', self)
        next_frame_5.setShortcut(QtCore.Qt.ALT + QtCore.Qt.Key_Right)
        next_frame_5.triggered.connect(self.next_frame_5)

        prev_frame_5 = QtWidgets.QAction('Back 5', self)
        prev_frame_5.setShortcut(QtCore.Qt.ALT + QtCore.Qt.Key_Left)
        prev_frame_5.triggered.connect(self.prev_frame_5)

        next_frame_50 = QtWidgets.QAction('Forward 50', self)
        next_frame_50.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_Right)
        next_frame_50.triggered.connect(self.next_frame_50)

        prev_frame_50 = QtWidgets.QAction('Back 50', self)
        prev_frame_50.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_Left)
        prev_frame_50.triggered.connect(self.prev_frame_50)

        next_frame_mar = QtWidgets.QAction('Next marked', self)
        next_frame_mar.setShortcut(QtCore.Qt.SHIFT + QtCore.Qt.Key_Right)
        next_frame_mar.triggered.connect(self.next_frame_mark)

        prev_frame_mar = QtWidgets.QAction('Previous marked', self)
        prev_frame_mar.setShortcut(QtCore.Qt.SHIFT + QtCore.Qt.Key_Left)
        prev_frame_mar.triggered.connect(self.prev_frame_mark)

        next_frame_mid = QtWidgets.QAction('Next midway', self)
        next_frame_mid.setShortcut(QtCore.Qt.SHIFT + QtCore.Qt.CTRL + QtCore.Qt.Key_Right)
        next_frame_mid.triggered.connect(self.next_frame_midpoint)

        prev_frame_mid = QtWidgets.QAction('Previous midway', self)
        prev_frame_mid.setShortcut(QtCore.Qt.SHIFT + QtCore.Qt.CTRL + QtCore.Qt.Key_Left)
        prev_frame_mid.triggered.connect(self.prev_frame_midpoint)

        nav_menu.addActions([next_frame, prev_frame])
        nav_menu.addSeparator()
        nav_menu.addActions([next_frame_5, prev_frame_5, next_frame_50, prev_frame_50])
        nav_menu.addSeparator()
        nav_menu.addActions([next_frame_mar, prev_frame_mar, next_frame_mid, prev_frame_mid])

        #  calibration menu
        cal_menu = menubar.addMenu('Calibration')

        camera_geometry_action = QtWidgets.QAction('Camera geometry...', self)
        camera_geometry_action.triggered.connect(self.open_board_calibration_dialog)
        cal_menu.addAction(camera_geometry_action)

        load_camera_action = QtWidgets.QAction('Load camera geometry...', self)
        load_camera_action.setShortcut(QtCore.Qt.SHIFT + QtCore.Qt.CTRL + QtCore.Qt.Key_C)
        load_camera_action.triggered.connect(self.load_checkerboard)
        cal_menu.addAction(load_camera_action)

        save_camera_action = QtWidgets.QAction('Save camera geometry...', self)
        save_camera_action.triggered.connect(self.save_checkerboard)
        cal_menu.addAction(save_camera_action)

        cal_menu.addSeparator()

        orientation_action = QtWidgets.QAction('Orientation of gravity...', self)
        orientation_action.triggered.connect(self.open_orientation_dialog)
        cal_menu.addAction(orientation_action)

        load_orientation_action = QtWidgets.QAction('Load orientation...', self)
        load_orientation_action.setShortcut(QtCore.Qt.SHIFT + QtCore.Qt.CTRL + QtCore.Qt.Key_P)
        load_orientation_action.triggered.connect(self.load_plumbline)
        cal_menu.addAction(load_orientation_action)

        save_orientation_action = QtWidgets.QAction('Save orientation...', self)
        save_orientation_action.triggered.connect(self.save_plumbline)
        cal_menu.addAction(save_orientation_action)

        #  markers menu
        mark_menu = menubar.addMenu('Markers')
        marker_group = QtWidgets.QActionGroup(self)

        actions = []
        for n in range(9):
            action = QtWidgets.QWidgetAction(self, checkable=True)
            radio = QtWidgets.QRadioButton(f'Marker {n + 1}', self)
            color = QtGui.QColor(*colors[n])
            radio.setStyleSheet(f'color: {color.name()}')
            action.setDefaultWidget(radio)
            action.setActionGroup(marker_group)
            action.triggered.connect(self.choose_marker)
            radio.clicked.connect(self.choose_marker)
            actions.append(action)

        mark_menu.addActions(actions)

        #  help menu
        help_menu = menubar.addMenu('Help')
        help_action = QtWidgets.QAction('Print help', self)
        help_action.triggered.connect(self.print_help)
        help_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_H)

        help_menu.addAction(help_action)

        # Create the QVBoxLayout that lays out the whole form
        w = QtWidgets.QWidget()
        self.layout = QtWidgets.QVBoxLayout(w)
        self.setCentralWidget(w)

        ### top row
        self.frame_hbox = QtWidgets.QHBoxLayout()

        # frame slider
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self.change_frame)
        # self.frame_value = pg.ValueLabel()
        self.frame_value = QtWidgets.QLabel('0/0')
        self.frame_hbox.addWidget(self.frame_slider)
        self.frame_hbox.addWidget(self.frame_value)

        ### middle row
        self.images_hbox = QtWidgets.QHBoxLayout()

        ### camera imageviews
        self.ims = [Imframe(self.images_hbox, self), Imframe(self.images_hbox, self)]
        # shortcuts
        self.l = self.ims[0]
        self.r = self.ims[1]

        ### bottom row
        ### 3d row
        self.three_d_hbox = QtWidgets.QHBoxLayout()

        vbox = QtWidgets.QVBoxLayout()
        namespace = {'pg': pg, 'np': np, 'st': self}
        text = 'st for stereography window\npg for pyqtgraph\nnp for numpy\n\n'
        self.console = Console(self, namespace, text)
        self.console.setMinimumWidth(500)
        vbox.addWidget(self.console)

        self.three_d_hbox.addItem(vbox)

        # 3d plot
        self.td = TDframe(self.three_d_hbox, self.ims)

        # marker location table and calibration buttons
        vbox = QtWidgets.QVBoxLayout()

        # marker location table
        self.tab = QtWidgets.QLabel('')
        self.tab.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        self.tab.setFont(QtGui.QFont('Monospace', 12))
        self.tab.setWordWrap(True)
        self.tab.setMinimumWidth(300)
        self.tab.setMaximumWidth(310)
        vbox.addWidget(self.tab)
        self.update_table()
        self.three_d_hbox.addItem(vbox)

        # load calibration buttons
        hindicators = QtWidgets.QHBoxLayout()
        # checkerboard calibration button
        self.chk_indicator = QtWidgets.QPushButton(text='camera', parent=self)
        self.chk_indicator.clicked.connect(self.load_checkerboard)
        self.chk_indicator.setMinimumHeight(25)
        self.chk_indicator.setMaximumHeight(30)
        self.set_calibration_indicator('chk', 'None')
        # plumbline calibration button
        self.pb_indicator = QtWidgets.QPushButton(text='orientation', parent=self)
        self.pb_indicator.clicked.connect(self.load_plumbline)
        self.pb_indicator.setMinimumHeight(25)
        self.pb_indicator.setMaximumHeight(30)
        self.set_calibration_indicator('pb', 'None')
        hindicators.addWidget(self.chk_indicator)
        hindicators.addWidget(self.pb_indicator)
        vbox.addLayout(hindicators)

        ### set the layout
        self.layout.addLayout(self.frame_hbox)
        self.layout.addLayout(self.images_hbox)
        self.layout.addLayout(self.three_d_hbox)

        ### now the markers
        self.num_frames = 1  # until we load a dir
        self.marker_keys = [QtCore.Qt.Key.Key_1, QtCore.Qt.Key.Key_2, QtCore.Qt.Key.Key_3, QtCore.Qt.Key.Key_4,
                            QtCore.Qt.Key.Key_5, QtCore.Qt.Key.Key_6, QtCore.Qt.Key.Key_7, QtCore.Qt.Key.Key_8,
                            QtCore.Qt.Key.Key_9]

        # print the help text
        self.print_help()

        ### double control action flags
        self.ctrl_x = False
        self.ctrl_c = False

        ### keypress actions
        self.key_actions = {
            # prefixes
            ('control C'): (self.set_ctrl_action, ('C',)),
            ('control X'): (self.set_ctrl_action, ('X',)),
            # undo
            ('control Z'): (self.undo, ()),
            # show help
            # ('control H')           : (self.set_info, (self.help_text,)),
            # arrows to shift frame
            # ('left')                : (self.move_frame_slider, ( -1,)),
            # ('right')               : (self.move_frame_slider, (  1,)),
            # ('control left')        : (self.move_frame_slider, (-50,)),
            # ('control right')       : (self.move_frame_slider, ( 50,)),
            # ('shift left')          : (self.move_frame_slider, (  0,-1)),
            # ('shift right')         : (self.move_frame_slider, (  0, 1)),
            # backquote resets the view
            ('`'): (self.reset_view, ()),
            # add markers
            ('1'): (self.move_marker, (None,)),
            ('2'): (self.move_marker, (None,)),
            ('3'): (self.move_marker, (None,)),
            ('4'): (self.move_marker, (None,)),
            ('5'): (self.move_marker, (None,)),
            ('6'): (self.move_marker, (None,)),
            ('7'): (self.move_marker, (None,)),
            ('8'): (self.move_marker, (None,)),
            ('9'): (self.move_marker, (None,)),
            # remove markers
            ('alt 1'): (self.move_marker, (None, False)),
            ('alt 2'): (self.move_marker, (None, False)),
            ('alt 3'): (self.move_marker, (None, False)),
            ('alt 4'): (self.move_marker, (None, False)),
            ('alt 5'): (self.move_marker, (None, False)),
            ('alt 6'): (self.move_marker, (None, False)),
            ('alt 7'): (self.move_marker, (None, False)),
            ('alt 8'): (self.move_marker, (None, False)),
            ('alt 9'): (self.move_marker, (None, False)),
            # add marker, then advance 50 frames
            ('control 1'): (self.move_marker, (None, True, None, 50)),
            ('control 2'): (self.move_marker, (None, True, None, 50)),
            ('control 3'): (self.move_marker, (None, True, None, 50)),
            ('control 4'): (self.move_marker, (None, True, None, 50)),
            ('control 5'): (self.move_marker, (None, True, None, 50)),
            ('control 6'): (self.move_marker, (None, True, None, 50)),
            ('control 7'): (self.move_marker, (None, True, None, 50)),
            ('control 8'): (self.move_marker, (None, True, None, 50)),
            ('control 9'): (self.move_marker, (None, True, None, 50)),
            # fullscreen toggle
            ('control F'): (self.toggle_fullscreen, ())
        }

        # fresh undo list
        self.undo_list = []

        self.fullscreen = False

        # the window takes the keypresses
        self.grabKeyboard()
        self.setMouseTracking(True)

    def load_avis(self, fn=None):
        '''Load all corresponding avis into Imframes, then get the calibration
        and the plumbline, if available, and load data if it exists.

        '''
        # use a dialog if no fn
        if not fn or not os.path.isfile(fn):
            fn = str(QtWidgets.QFileDialog.getOpenFileName(self, 'Select video', '', 'avi files (*.avi)')[0])
        d, f = os.path.split(fn)

        # look for the corresponding file, starting with L or R
        if f.startswith('L') and os.path.isfile(os.path.join(d, f'R{f[1:]}')):
            self.fns = [os.path.join(d, f), os.path.join(d, f'R{f[1:]}')]

        elif f.startswith('R') and os.path.isfile(os.path.join(d, f'L{f[1:]}')):
            self.fns = [os.path.join(d, f'L{f[1:]}'), os.path.join(d, f)]

        # or if fn dialog was cancelled
        elif f == '':
            return

        # or open a dialog to get the other file
        else:
            # self.fns = [None, None]
            fn = str(QtWidgets.QFileDialog.getOpenFileName(self, 'Select other video', '', 'avi files (*.avi)')[0])
            d2, f2 = os.path.split(fn)

            # if dialog was cancelled
            if f2 == '':
                return
            # otherwise this is the second
            else:
                self.fns = [os.path.join(d, f), os.path.join(d2, f2)]

        # load the images
        for im, avi in zip(self.ims, self.fns):
            im.load_avi(avi)

        # how many frames?
        self.num_frames = self.ims[0].num_frames
        self.frame_slider.setMaximum(self.num_frames - 1)
        self.frame_slider.setValue(0)
        self.change_frame(0)

        # set blank data
        data = [None] * (len(self.ims) + 1)
        for i, im in enumerate(self.ims):
            im.set_data(data[i])
        # self.td.set_data(data[-1])
        self.td.set_data()

        self.data_fn = ''
        self.npy_fn = ''
        self.csv_fn = ''

        title_fns = self.fns.copy()
        # get data from txt file:
        for i, fn in enumerate(title_fns):
            fn_txt = fn[:-3] + 'txt'
            if os.path.isfile(fn_txt):
                f = open(fn_txt, 'r')
                lns = f.readlines()
                f.close()

                lns = [ln.strip() for ln in lns]
                fps = [int(l.split('=')[1]) for l in lns if l.startswith('fps')][0]
                ts = [l.split('=')[1] for l in lns if l.startswith('time_stamp')][0]

                title_fns[i] = f'{title_fns[i]} {":".join(ts.split(":")[2:4])} {fps} fps'

        self.setWindowTitle(f'{self.window_title}: {"      &      ".join([os.path.split(fn)[-1] for fn in title_fns])}')

        for im in self.ims:
            im.set_frame(0, True, True)
            im.show_markers()

        self.console_write(f'{self.fns[0]}\n{self.fns[1]}', 'loaded AVIs')

    def set_chessboard_size(self, board_rows, board_cols):
        '''Set the calibration rows and columns. The inner squares are one
        less than the board sizes

        '''
        self.cal_rows = board_rows - 1
        self.cal_cols = board_cols - 1
        print(f'set_chessboard_size {self.cal_rows} x {self.cal_cols}')

    def test_aruco_dictionaries(self):
        frame_ind = self.frame_slider.value()

        dictionaries = [
            ("4X4_50", cv.aruco.DICT_4X4_50),
            ("4X4_100", cv.aruco.DICT_4X4_100),
            ("4X4_250", cv.aruco.DICT_4X4_250),
            ("5X5_50", cv.aruco.DICT_5X5_50),
            ("5X5_100", cv.aruco.DICT_5X5_100),
            ("5X5_250", cv.aruco.DICT_5X5_250),
            ("6X6_50", cv.aruco.DICT_6X6_50),
            ("6X6_100", cv.aruco.DICT_6X6_100),
            ("6X6_250", cv.aruco.DICT_6X6_250),
            ("7X7_50", cv.aruco.DICT_7X7_50),
            ("7X7_100", cv.aruco.DICT_7X7_100),
            ("7X7_250", cv.aruco.DICT_7X7_250),
        ]

        for cam_ind, im in enumerate(self.ims):
            im.cap.set(cv.CAP_PROP_POS_FRAMES, frame_ind)
            ok, frame = im.cap.read()

            if not ok:
                print(f"camera {cam_ind}: could not read frame {frame_ind}")
                continue

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            print(f"\ncamera {cam_ind}, frame {frame_ind}")
            for name, dict_id in dictionaries:
                dictionary = cv.aruco.getPredefinedDictionary(dict_id)
                detector = cv.aruco.ArucoDetector(dictionary)
                corners, ids, rejected = detector.detectMarkers(gray)
                n = 0 if ids is None else len(ids)
                print(f"{name}: {n} markers")

    def open_board_calibration_dialog(self):
        """Open the unified board-calibration dialog."""
        dialog = BoardCalibrationDialog(self, self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        settings = dialog.settings()
        if settings['board_type'] == 'charuco':
            self.get_charuco_board(settings=settings)
        elif settings['board_type'] == 'checkerboard':
            self.get_checkerboard(settings=settings)
        elif settings['board_type'] == 'circle_grid':
            self.get_circle_grid(settings=settings)
        else:
            self.console_write(
                'This board type is not implemented in the dialog yet.',
                'board calibration'
            )

    def open_orientation_dialog(self):
        """Open the orientation of gravity calibration dialog."""
        dialog = OrientationCalibrationDialog(self, self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        settings = dialog.settings()
        if settings['method'] == 'plumbline':
            self.get_plumbline()
        elif settings['method'] == 'projectile_center':
            self.get_projectile_orientation(
                marker_ind=settings['projectile_marker_ind'],
                fps=settings['projectile_fps']
            )
        else:
            self.console_write(
                'This orientation method is not implemented yet.',
                'orientation'
            )

    def _calibration_frame_indices(self, settings):
        """Return frame indices requested by the calibration dialog."""
        mode = settings.get('frame_mode', 'marked_or_even')
        max_frame = max(0, self.num_frames - 1)

        if mode == 'current':
            return np.array([int(self.frame_slider.value())], dtype=int)

        if mode == 'range':
            start = int(np.clip(settings.get('start_frame', 0), 0, max_frame))
            stop = int(np.clip(settings.get('stop_frame', max_frame), 0, max_frame))
            step = max(1, int(settings.get('frame_step', 1)))
            if stop < start:
                start, stop = stop, start
            return np.arange(start, stop + 1, step, dtype=int)

        if mode == 'even':
            num = max(1, int(settings.get('num_frames', 20)))
            return np.linspace(0, max_frame, num, dtype=int)

        # Default: use frames marked with marker 1 in either image. If there
        # are no marked frames, fall back to evenly spaced frames.
        inds = np.where(np.logical_or(*[im.data[0, -1] for im in self.ims]))[0]
        print(f'{inds=}')
        if len(inds) == 0:
            num = max(1, int(settings.get('num_frames', 20)))
            inds = np.linspace(0, max_frame, num, dtype=int)
            print('no inds marked')
        return inds.astype(int)

    def get_checkerboard(self, settings=None):
        """Search each imframe for checkerboard calibration images, calibrate
        both cameras, then construct stereo projection matrices.

        If settings comes from the board calibration dialog, squares_x and
        squares_y are board squares, so one is subtracted from each dimension
        to get the inner checkerboard intersections OpenCV expects.
        """
        if settings is None:
            inner_x = self.cal_rows
            inner_y = self.cal_cols
            side_len = self.cal_side
            inds = np.where(np.logical_or(*[im.data[0, -1] for im in self.ims]))[0]
            print(f'{inds=}')
            if len(inds) == 0:
                num = 20
                inds = np.linspace(0, self.num_frames - 1, num, dtype='int')
                print('no inds marked')
        else:
            inner_x = max(1, int(settings['squares_x']) - 1)
            inner_y = max(1, int(settings['squares_y']) - 1)
            side_len = float(settings['square_len'])
            inds = self._calibration_frame_indices(settings)

        print(f'{inds=}\n')
        ok = []
        for im in self.ims:
            ok.append(
                im.get_calibration(
                    inds,
                    num_rows=inner_x,
                    num_cols=inner_y,
                    side_len=side_len,
                    use_roi=True if settings is None else bool(settings.get('use_roi', True))
                )
            )

        if not all(ok):
            self.console_write("Checkerboard calibration failed in one or both cameras.", "calibration")
            return

        try:
            self.td.get_calibration()
        except Exception as e:
            self.console_write(f"Checkerboard stereo calibration failed: {e}", "calibration")
            return

        self.console_write(f'frames: {self.td.cal_inds}\nroot mean squared error = {self.td.rmse}', 'calibration')

    def get_circle_grid(self, settings=None):
        """Search each imframe for asymmetric circle-grid calibration images,
        calibrate both cameras, then construct stereo projection matrices.

        If settings comes from the board calibration dialog, squares_x and
        squares_y are interpreted directly as columns and rows of detected
        circle centers.
        """
        if settings is None:
            # Legacy direct-menu behavior.
            rows = self.cal_rows + 1
            cols = self.cal_cols + 1
            side_len = self.cal_side
            grid_type = 'asymmetric'
            first_row_count = 5
            second_row_count = 6
            inds = np.where(np.logical_or(*[im.data[0, -1] for im in self.ims]))[0]
            print(f'{inds=}')
            if len(inds) == 0:
                num = 20
                inds = np.linspace(0, self.num_frames - 1, num, dtype='int')
                print('no inds marked')
            use_roi = True
        else:
            cols = max(1, int(settings['squares_x']))
            rows = max(1, int(settings['squares_y']))
            side_len = float(settings['square_len'])
            grid_type = settings.get('circle_grid_type', 'asymmetric')
            first_row_count = settings.get('odd_col_count', settings.get('first_row_count', cols))
            second_row_count = settings.get('even_col_count', settings.get('second_row_count', cols))
            inds = self._calibration_frame_indices(settings)
            use_roi = bool(settings.get('use_roi', True))

        print(f'{inds=}\n')

        ok = []
        for im in self.ims:
            ok.append(
                im.get_circle_calibration(
                    inds,
                    num_rows=rows,
                    num_cols=cols,
                    side_len=side_len,
                    use_roi=use_roi,
                    grid_type=grid_type,
                    first_row_count=first_row_count,
                    second_row_count=second_row_count
                )
            )

        if not all(ok):
            self.console_write("Circle-grid calibration failed.", "calibration")
            return

        try:
            self.td.get_calibration()
        except Exception as e:
            self.console_write(f"Circle-grid stereo calibration failed: {e}", "calibration")
            return

        self.console_write(
            f'frames: {self.td.cal_inds}\nroot mean squared error = {self.td.rmse}',
            'calibration'
        )

    def get_charuco_board(self, settings=None):
        """Compute a stereo calibration from ChArUco board frames."""
        if settings is None:
            # Keep the old menu action working while the dialog is introduced.
            # The old board-size menu stores checkerboard inner-corner counts,
            # so add one to get ChArUco square counts.
            settings = {
                'squares_x': self.cal_cols + 1,
                'squares_y': self.cal_rows + 1,
                'square_len': self.cal_side,
                'marker_len': self.charuco_marker_len,
                'dictionary_id': self.charuco_dictionary,
                'frame_mode': 'marked_or_even',
                'num_frames': 20,
                'use_roi': True,
                'legacy_pattern': False,
            }

        rows = int(settings['squares_y'])
        cols = int(settings['squares_x'])
        inds = self._calibration_frame_indices(settings)

        print(f'{inds=}\n')

        ok = []
        for im in self.ims:
            ok.append(
                im.get_charuco_calibration(
                    inds,
                    num_rows=rows,
                    num_cols=cols,
                    side_len=float(settings['square_len']),
                    marker_len=float(settings['marker_len']),
                    dictionary_id=settings['dictionary_id'],
                    legacy_pattern=bool(settings.get('legacy_pattern', False)),
                    use_roi=bool(settings.get('use_roi', True))
                )
            )

        if not all(ok):
            self.console_write("ChArUco-board calibration failed.", "calibration")
            return

        try:
            self.td.get_calibration()
        except Exception as e:
            self.console_write(f"ChArUco stereo calibration failed: {e}", "calibration")
            return

        self.console_write(
            f'frames: {self.td.cal_inds}\nroot mean squared error = {self.td.rmse}',
            'calibration'
        )

    def get_plumbline(self, eye=False):
        '''Make a rotation matrix, assuming marker 1 and 2 are the top and
        bottom of a plumbline, that rotates downward in the correct
        direction.

        '''
        result = self.td.get_plumbline()
        if result:
            self.console_write(result, 'orientation')

        self.console_write(f'{self.td.pbrot}', 'orientation rotation')

    def get_projectile_orientation(self, marker_ind=0, fps=1.0):
        '''Orient the scene by fitting gravity from a projectile trajectory.'''
        if self.td.got_cal is None:
            self.console_write('Need camera geometry before thrown-object orientation.', 'orientation')
            return

        try:
            result = self.td.get_projectile_orientation(marker_ind=marker_ind, fps=fps)
        except Exception as e:
            self.console_write(f'Thrown-object orientation failed: {e}', 'orientation')
            return

        self.console_write(result, 'orientation')
        self.console_write(f'{self.td.pbrot}', 'orientation rotation')

    def set_info(self, help_text=''):
        '''Display filenames for images, calibration, distortion, and
        plumbline, data save, and the most recent help message.

        '''
        viddir = f'directory:   {self.dir}\n'
        vs = [os.path.split(im.fn)[-1] for im in self.ims]
        vids = 'videos:      ' + "\n             ".join(vs) + '\n'
        c = 'None' if self.checkerboard_fn is None else os.path.split(self.checkerboard_fn)[-1]
        chk = f'checkerboard: {c}\n'
        sn = os.path.split(self.data_fn)[-1]
        sv = f'saving to:   {sn}\n'
        text = '\n'.join([viddir, vids, chk, sv, '\n\n', help_text])
        # self.info.setText(text)
        self.console_write(text)

    def update_table(self):
        '''Make a table of valid marker values.

        '''
        text = ['<pre>M:   x       y       z</pre>']
        frame_ind = int(self.frame_slider.value())
        for marker_ind in range(self.num_markers):
            if frame_ind in self.td.get_valid_inds(marker_ind):
                x, y, z = self.td.data[marker_ind, :, frame_ind]
                r, g, b = colors[marker_ind]
                t = f"<pre style='color: rgb({r}, {g}, {b});'>{marker_ind + 1}: {x:7.01f} {y:7.01f} {z:7.01f}</pre>"
                text.append(t)
                # self.t = t

        self.tab.setText('\n'.join(text))

    def change_frame(self, frame_ind=None, autorange=False, autolevel=False):
        '''Changes to a new frame in the avi by updating the images, 3d view,
        and the table of marker position

        '''
        if frame_ind is None:
            frame_ind = self.frame_slider.value()
        self.frame_value.setText('{}/{}'.format(self.frame_slider.value(), self.num_frames - 1))

        # new image frame change
        for im in self.ims:
            im.set_frame(frame_ind, autorange, autolevel)
        self.td.set_frame(frame_ind)

        self.update_table()

    def reset_view(self):
        '''Reset the view. In an imframe this resets scaling and brightness
        ranges, in the three dimensional view, it pans the focus to
        camera 0.

        '''
        frame_ind = self.frame_slider.value()
        for im in self.ims:
            if im.imview.imageItem.isUnderMouse():
                im.set_frame(frame_ind, autorange=True, autolevel=True)

        if self.td.view.underMouse():
            self.td.pan(-1)

    def reset_views(self):
        '''Reset all the views no matter what is under the mouse

        '''
        frame_ind = self.frame_slider.value()
        for im in self.ims:
            im.set_frame(frame_ind, autorange=True, autolevel=True)
        self.td.pan(-1)

    def ims_quantile(self, qmin, qmax):
        '''Change the levels of all images to constrict the brightness
        range

        '''
        for im in self.ims:
            im.adjust_levels_quantile(qmin, qmax)

    # functions calling ims_quantile with useful defaults
    def adjust_lower_quantile(self):
        self.ims_quantile(qmin=0.5, qmax=1.0)

    def adjust_middle_quantile(self):
        self.ims_quantile(qmin=0.15, qmax=0.85)

    def adjust_upper_quantile(self):
        self.ims_quantile(qmin=0.0, qmax=0.5)

    # def move_frame_slider(self, n=0, next_marked=None):
    #     '''Move the frame slider n frames. If next_marked is not 0, n to
    #     the next or previous marked frame. For all the marked values
    #     from all camera images, for all markers, return the distance
    #     to the next frame marked with any marker.

    #     '''
    #     # if next_marked, calculate the next marked frame and set n
    #     if next_marked is not None:
    #         current = self.frame_slider.value()
    #         # select marked frames from both cameras
    #         marked = np.concatenate([np.where(im.data[:,-1]==1)[1] for im in self.ims])
    #         # how far are we now
    #         dists = marked-current
    #         if next_marked > 0 and dists[dists>0].size > 0:
    #             n = dists[dists>0].min()
    #         elif next_marked < 0 and dists[dists<0].size > 0:
    #             n = dists[dists<0].max()
    #         else:
    #             n = 0

    #     # otherwise just add n and clip it to a valid value
    #     new_value = np.clip(self.frame_slider.value() + n, 0, self.frame_slider.maximum())

    #     self.frame_slider.setValue(new_value)

    def _marked_frames(self, include_calibration=False):
        """Return sorted unique frame indices marked in any image.

        By default this uses only user-placed tracking markers. When
        include_calibration is true, frames where a calibration board was
        detected are also included so Shift+arrow can step through them for
        inspection.
        """
        marked_frames = []

        for im in self.ims:
            where_marked = np.where(im.data[:, -1] == 1)

            if len(where_marked) > 1:
                marked_frames.append(where_marked[1])
            else:
                marked_frames.append(where_marked[0])

            if include_calibration and getattr(im, 'cal_inds', None):
                marked_frames.append(np.asarray(im.cal_inds, dtype=int))

        marked_frames = [m for m in marked_frames if len(m) > 0]
        if not marked_frames:
            return np.array([], dtype=int)

        marked = np.concatenate(marked_frames)
        return np.unique(marked)

    def _frame_step_to_target(self, targets, direction):
        """Return offset from current frame to next target in the given direction."""
        current = self.frame_slider.value()
        targets = np.asarray(targets, dtype=int)
        dists = targets - current

        if direction > 0:
            ahead = dists[dists > 0]
            return ahead.min() if ahead.size else 0

        if direction < 0:
            behind = dists[dists < 0]
            return behind.max() if behind.size else 0

        return 0

    def move_frame_slider(self, n=0, next_marked=None, next_midway=None):
        """
        Move the frame slider by n frames, to the next marked frame,
        or to the next midpoint between adjacent marked frames.
        """

        if next_marked is not None and next_midway is not None:
            raise ValueError("Use either next_marked or next_midway, not both.")

        current = self.frame_slider.value()

        if next_marked is not None:
            marked = self._marked_frames(include_calibration=True)
            n = self._frame_step_to_target(marked, next_marked)

        elif next_midway is not None:
            marked = self._marked_frames()

            if marked.size >= 2:
                midpoints = (marked[:-1] + marked[1:]) // 2
                midpoints = np.unique(midpoints)
                n = self._frame_step_to_target(midpoints, next_midway)
            else:
                n = 0

        new_value = np.clip(
            current + n,
            0,
            self.frame_slider.maximum()
        )

        self.frame_slider.setValue(int(new_value))

    def next_frame(self):
        self.move_frame_slider(1)

    def prev_frame(self):
        self.move_frame_slider(-1)

    def next_frame_5(self):
        self.move_frame_slider(5)

    def prev_frame_5(self):
        self.move_frame_slider(-5)

    def next_frame_10(self):
        self.move_frame_slider(10)

    def prev_frame_10(self):
        self.move_frame_slider(-10)

    def next_frame_50(self):
        self.move_frame_slider(50)

    def prev_frame_50(self):
        self.move_frame_slider(-50)

    def next_frame_mark(self):
        self.move_frame_slider(0, next_marked=1)

    def prev_frame_mark(self):
        self.move_frame_slider(0, next_marked=-1)

    def next_frame_midpoint(self):
        self.move_frame_slider(0, next_midway=1)

    def prev_frame_midpoint(self):
        self.move_frame_slider(0, next_midway=-1)

    def get_keystroke(self):
        '''Turns the keystroke event into a readable dictionary key.

        '''
        # prefix, if there is one
        keystroke = []
        if self.ctrl_x:
            keystroke.append('control X')
            self.ctrl_x = False
        if self.ctrl_c:
            keystroke.append('control C')
            self.ctrl_c = False

        # modifiers
        m = self.key_event.modifiers()
        if (m & QtCore.Qt.KeyboardModifier.ShiftModifier) == QtCore.Qt.KeyboardModifier.ShiftModifier:
            keystroke.append('shift')
        if (m & QtCore.Qt.KeyboardModifier.ControlModifier) == QtCore.Qt.KeyboardModifier.ControlModifier:
            keystroke.append('control')
        if (m & QtCore.Qt.KeyboardModifier.AltModifier) == QtCore.Qt.KeyboardModifier.AltModifier:
            keystroke.append('alt')

        # key
        k = self.key_event.key()
        if 0 <= k <= 0x10ffff:
            keystroke.append(chr(k))

        elif k == QtCore.Qt.Key.Key_Left:
            keystroke.append('left')
        elif k == QtCore.Qt.Key.Key_Right:
            keystroke.append('right')
        elif k == QtCore.Qt.Key.Key_Up:
            keystroke.append('up')
        elif k == QtCore.Qt.Key.Key_Down:
            keystroke.append('down')

        else:
            keystroke.append('')

        return ' '.join(keystroke)

    def set_ctrl_action(self, key):
        '''Sets a state after pressing the prefix key of a key sequence. Right
        now control x and control c both expect another key to
        complete.

        '''
        if key == 'X':
            self.ctrl_x = True
        elif key == 'C':
            self.ctrl_c = True

    def keyPressEvent(self, event):
        '''Process a keystroke by matching to a dict key, and executing the
        function it pairs to

        '''
        # first check if the mouse is over the console:
        self.key_event = event
        dictkey = self.get_keystroke()

        if dictkey in self.key_actions:
            fun, args = self.key_actions[dictkey]
            fun(*args)

        # elif event.key() == QtCore.Qt.Key.Key_C:
        #     self.load_avis('/home/jamie/data/3d/test_t8/LTS5_000000_cal.avi')
        # elif event.key() == QtCore.Qt.Key.Key_L:
        #     self.load_avis('/home/jamie/data/3d/test3/LTS5_000002.avi')
        # elif event.key() == QtCore.Qt.Key.Key_T:
        #     self.load_avis('/home/jamie/data/3d/test3/LTS5_000003.avi')
        # elif event.key() == QtCore.Qt.Key.Key_P:
        #     self.load_avis('/home/jamie/data/3d/test3/LTS5_000001.avi')

    def move_marker(self, event=None, add=True, marker_ind=None, advance=0):
        '''Moves or removes a marker based pressing a number key with the
        mouse in a given position over an imframe. If the mouse is
        over the three-dimensional view, this sets the focus on the
        marker.

        '''
        if event is None:
            event = self.key_event
        if marker_ind is None:
            marker_ind = self.marker_keys.index(event.key())

        for im in self.ims:
            if im.imview.imageItem.isUnderMouse():
                im.set_marker(marker_ind, add=add)

        if self.td.view.underMouse():
            self.td.pan(marker_ind)

        self.camera_to_3d(marker_ind=marker_ind)

        # Optionally move on after marking. This supports shortcuts like
        # Ctrl+number, which mark a point and jump ahead to the next interval.
        if advance != 0:
            self.move_frame_slider(advance)

    def camera_to_3d(self, marker_ind):
        '''Redo the interpolation for a marker, and if there is a calibration
        file, update the 3D display data as well

        '''
        for im in self.ims:
            im.make_interp(marker_ind)
        if self.chk_calibration != 'None':
            self.td.update_data(marker_ind)
            self.td.show_lines()

    def save_data_as(self):
        '''Save the marker and 3d data with a dialog to get the filename.

        '''
        self.get_data_fn = True

        self.save_data()

    def save_data(self, name=None):
        '''Saves the marker data. If a filename exists, it uses that,
        otherwise opens a dialog.

        '''
        data = np.stack((*[im.data for im in self.ims],))

        if not self.data_fn or self.get_data_fn:

            self.get_data_fn = False

            ddir, dname = os.path.split(self.fns[0])
            if dname[0] in 'LR': dname = dname[1:]
            dfn = f'2D{dname[:-3]}npy'
            ddefaultname = os.path.join(ddir, dfn)

            data_fn = str(
                QtWidgets.QFileDialog.getSaveFileName(self, 'Save Data As', ddefaultname, 'npy files (*.npy)')[0])
            if data_fn:
                self.data_fn = data_fn

        if self.data_fn:
            np.save(self.data_fn, data)

            self.console_write(self.data_fn, 'save marker data')

        else:
            self.console_write('no save filename')

    def export_npy_as(self):
        '''Save the marker and 3d data with a dialog to get the filename.

        '''
        self.get_npy_fn = True

        self.export_npy()

    def export_npy(self):
        '''Export the 3d data into a npy file

        '''
        if not self.npy_fn or self.get_npy_fn:

            self.get_npy_fn = False

            ddir, dname = os.path.split(self.fns[0])
            if dname[0] in 'LR': dname = dname[1:]
            dfn = f'3D{dname[:-3]}npy'
            ddefaultname = os.path.join(ddir, dfn)

            npy_fn = str(
                QtWidgets.QFileDialog.getSaveFileName(self, 'Save Data As', ddefaultname, 'npy files (*.npy)')[0])
            if npy_fn:
                self.npy_fn = npy_fn

        if self.npy_fn:
            np.save(self.npy_fn, self.td.data)

            self.console_write(self.npy_fn, 'save marker data')

        else:
            self.console_write('no save filename')

    def export_csv_as(self):
        '''Save the marker and 3d data with a dialog to get the filename.

        '''
        self.get_csv_fn = True

        self.export_csv()

    def export_csv(self):
        '''Export the 3d data into a csv file

        '''
        if not self.csv_fn or self.get_csv_fn:

            self.get_csv_fn = False

            ddir, dname = os.path.split(self.fns[0])
            if dname[0] in 'LR': dname = dname[1:]
            dfn = f'3D{dname[:-3]}csv'
            ddefaultname = os.path.join(ddir, dfn)

            csv_fn = str(
                QtWidgets.QFileDialog.getSaveFileName(self, 'Save Data As', ddefaultname, 'csv files (*.csv)')[0])
            if csv_fn:
                self.csv_fn = csv_fn

        if self.csv_fn:
            np.savetxt(self.csv_fn, self.td.data, delimiter=',')

            self.console_write(self.csv_fn, 'save marker data')

        else:
            self.console_write('no save filename')

    def load_data(self):
        '''Loads the marker data.

        '''
        fn = str(QtWidgets.QFileDialog.getOpenFileName(self, 'Select data file', '', 'npy files (*.npy)')[0])

        if fn == '':
            return

        try:
            data = np.load(fn)

        except Exception as e:
            self.console_write(f"An error occurred: {e}")
            return

        # did we actually load the correct type of npy file?
        if len(data.shape) != 4 or data.shape[3] != self.ims[0].num_frames:
            self.console_write(fn, "data lengsh doesn't match")

        # if so then populate the images and three d plot
        else:
            self.data_fn = fn

            # fill in the image data
            for i, im in enumerate(self.ims):
                im.set_data(data[i])

            # blank the three-D data
            self.td.set_data()

            # update 3d marker positions, if we have a calibration
            for marker_ind in range(self.num_markers):
                self.camera_to_3d(marker_ind)

            self.console_write(fn, 'loaded marker data')

    def save_checkerboard(self, fn=None):
        '''Saves all the parameters of a stereo calibration---the matrixes for
        each camera. This has to mirror the load_calibration method.

        '''

        if self.ims[0].fn == '' or self.td.got_cal is None:
            self.console_write('Need camera geometry first')
            return

        # get the directory and file name
        cdir, cname = os.path.split(self.fns[0])
        # if we start with L or R, drop that for the save name
        if cname.startswith('L') or cname.startswith('R'):
            cname = cname[1:]
        # get rid of the .avi
        cname = cname.rsplit('.', 1)[0]
        cname = f'C{cname}.npy'
        cdefaultname = os.path.join(cdir, cname)

        fn = str(
            QtWidgets.QFileDialog.getSaveFileName(self, 'Save camera geometry', cdefaultname, 'npy files (*.npy)')[0])

        data = np.zeros((27, 5))

        # camera matrixes
        data[0:3, 0:3] = self.ims[0].mtx
        data[3:6, 0:3] = self.ims[1].mtx
        data[6:9, 0:3] = self.ims[0].nmtx
        data[9:12, 0:3] = self.ims[1].nmtx
        # projection matrixes
        data[12:15, 0:4] = self.ims[0].proj
        data[15:18, 0:4] = self.ims[1].proj
        # distortion coefficients
        data[18:19, 0:5] = self.ims[0].dist
        data[19:20, 0:5] = self.ims[1].dist
        # R matrix
        data[20:23, 0:3] = self.td.R
        # T matrix
        data[23:26, 0:1] = self.td.T

        np.save(fn, data)

        self.set_calibration_indicator('chk', fn=fn)

        self.console_write(fn, 'saved camera geometry')

    def save_plumbline(self, fn=None):
        '''Save the plumbline rotation matrix, which points the scene down.

        '''
        if self.ims[0].fn == '' or self.td.got_pb is None:
            self.console_write('Need orientation first')
            return

        # get the directory and file name
        pdir, pname = os.path.split(self.fns[0])
        # if we start with L or R, drop that for the save name
        if pname.startswith('L') or pname.startswith('R'):
            pname = pname[1:]
        # get rid of the .avi
        pname = pname.rsplit('.', 1)[0]
        pname = f'P{pname}.npy'
        pdefaultname = os.path.join(pdir, pname)

        fn = str(QtWidgets.QFileDialog.getSaveFileName(self, 'Save orientation', pdefaultname, 'npy files (*.npy)')[0])

        np.save(fn, self.td.pbrot)

        self.set_calibration_indicator('pb', fn=fn)

        self.console_write(fn, 'saved orientation')

    def load_checkerboard(self, fn=None):
        '''Get the checkerboard calibration.

        '''
        fn = str(QtWidgets.QFileDialog.getOpenFileName(self, 'Select camera geometry', '', 'npy files (*.npy)')[0])

        if fn == '':
            return

        try:
            data = np.load(fn)
            assert data.shape == (27, 5)

        except Exception as e:
            self.console_write(f"An error occurred: {e}")
            return

        # camera matrixes
        self.ims[0].mtx = data[0:3, 0:3]
        self.ims[1].mtx = data[3:6, 0:3]
        self.ims[0].nmtx = data[6:9, 0:3]
        self.ims[1].nmtx = data[9:12, 0:3]
        # projection matrixes
        self.ims[0].proj = data[12:15, 0:4]
        self.ims[1].proj = data[15:18, 0:4]
        # distortion coefficients
        self.ims[0].dist = data[18:19, 0:5]
        self.ims[1].dist = data[19:20, 0:5]
        # R matrix
        self.td.R = data[20:23, 0:3]
        # T matrix
        self.td.T = data[23:26, 0:1]
        self.td.got_cal = True
        self.td.set_camera_markers()

        self.set_calibration_indicator('chk', fn=fn)

        self.console_write(fn, f'loaded camera geometry')

    def load_plumbline(self, fn=None):
        '''Get the plumbline calibration.

        '''
        fn = str(QtWidgets.QFileDialog.getOpenFileName(self, 'Select orientation', '', 'npy files (*.npy)')[0])

        if fn == '':
            return

        try:
            data = np.load(fn)
            assert data.shape == (3, 3)

        except Exception as e:
            self.console_write(f"An error occurred: {e}")
            return

        self.td.pbrot = data
        self.td.got_pb = True

        self.set_calibration_indicator('pb', fn=fn)

        self.console_write(fn, f'loaded orientation')

    def set_calibration_indicator(self, cal, fn='None'):
        '''Change the color and hover text of the calibration indicators,
        checkerboard and plumbline.

        '''
        if cal.startswith('chk'):
            label = self.chk_indicator
            self.chk_calibration = fn
        elif cal.startswith('pb'):
            label = self.pb_indicator
            self.pb_calibration = fn
        if fn == 'None':
            color = QtGui.QColor(196, 79, 81, 255)
        else:
            color = QtGui.QColor(84, 168, 104, 255)

        # label.setStyleSheet(f'background-color: {color.name()};')
        label.setStyleSheet(f'color: {color.name()}; font-size: 18px;')
        label.setToolTip(fn)

    def choose_marker(self, arg=0):
        '''Choose the current active marker for clicks

        '''
        self.action = self.sender()
        self.curr_marker = int(self.action.text()[-1])
        self.console_write(f'marker {self.curr_marker}')

    def console_write(self, info, subject=None):
        '''Write to the console with nicely formatted data, adding newlines
        after each entry, and an underlined subject if requested

        '''
        out = f'{info}\n\n'
        if subject is not None:
            out = f'{subject.capitalize()}:\n{"=" * (len(subject) + 1)}\n{out}'
        self.console.write(out)

    def undo(self):
        '''Pops the command off the end of the undo list, then executes
        it. The command includes the imframe it came from and the
        marker state. Because changing the marker state adds the
        previous state to the undo list, we pop that off, so iterative
        undo goes back down the list, not back and forth between the
        two most recent states

        '''
        if len(self.undo_list) > 0:
            # get last state and execute the command to retrieve it
            fun, args = self.undo_list.pop()
            fun(*args)

            # purge the new last state (what we just undid)
            self.undo_list.pop()

        # update the 3d plots as if a marker moved
        for marker_ind in range(self.num_markers):
            self.camera_to_3d(marker_ind)

    def toggle_fullscreen(self):
        if self.fullscreen:
            self.showNormal()
            self.fullscreen = False
        else:
            self.showFullScreen()
            self.fullscreen = True

    def enterEvent(self, event):
        self.grabKeyboard()

    def leaveEvent(self, event):
        self.releaseKeyboard()

    def run(self):
        '''Runs the app.

        '''
        self.show()
        pg.QtWidgets.QApplication.exec()

    def print_help(self):
        help_text = '\n'.join([
            'control h:               display this help text',
            'control f:               toggle fullscreen',
            ' ',
            'left and right arrows:   step through frames',
            'control left and right:  step +-50 frames',
            'shift left and right:    step through marked',
            ' ',
            '1-9 (image):             add a marker',
            'control 1-9 (image):     add marker and advance 50 frames',
            ' ',
            'alt 1-9 (image):         remove the marker',
            ' ',
            '1-9 (3D display)         pan to the marker',
            ' ',
            'control z:               undo',
            ' ',
            '` (backquote):           reset the image views'])

        # self.set_info(self.help_text)
        self.console_write(help_text, 'key commands')


qt_app = QtWidgets.QApplication(sys.argv)
s = Stereography_window(cal_rows=7, cal_cols=6, cal_side=11.5)
s.run()

# def scale(arr, mn, mx):
#     a = np.interp(arr, [mn, mx], [0,255])
#     return a.astype('uint8')

# iml=s.l.image
# imr=s.r.image
# fc = cv.findChessboardCorners
# fl = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
# n = (7,6)
