#! /usr/bin/env python

'''Stereography window uses pyqtgraph and opencv to mark features from
two camera views and image the three dimensional reconstruction of
their paths simultaneously.

'''

# update with drag pt

import numpy as np
import cv2 as cv
import os
import sys

import time
import datetime
from string import printable

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
    r = np.clip(-57.4*np.sin(y + 4.5) +x*265.1 + -20.3, 0, 255)
    g = np.clip(20.6*np.sin(y + -1.5) +x*253.5 + -14.3, 0, 255)
    b = np.clip(57.1*np.sin(y + 3.2) +x*234.6 + -4.3, 0, 255)
    return np.stack([int(r), int(g), int(b)])

# make a ring of colors with the same darkness
colors = [cm_soapbubble(.5, y) for y in np.linspace(0, 2*np.pi, 9, endpoint=False)]
# mix the colors so near ones aren't adjacent
mix = ((np.arange(9) * 2)% 9)
# start with blue and green
mix = np.roll(mix[::-1], 4)
colors = [cm_soapbubble(.6, y) for y in mix/9. * 2 * np.pi]


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
        print(f"Selected board size: {row+1}x{col+1}")
        # print(f"{dir(self.parent())}")
        self.parent().close()
        self.st_win.set_chessboard_size(row+1, col+1)



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
        #pyqtgraph's opengl 3d viewer
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
        self.gz = gl.GLGridItem(color=pg.mkColor([76, 114, 175,255]))
        self.gz.translate(0, 0, -10)
        self.view.addItem(self.gz)

        self.ax = gl.GLAxisItem()
        self.view.addItem(self.ax)

        self.curr_pan = np.array([-1,0,0,0])
        self.focus = np.array([0,0,0])

        self.c1 = gl.GLTextItem()
        self.c1.setData(text='C1')
        self.view.addItem(self.c1)
        
        self.c2 = gl.GLTextItem()
        self.c2.setData(text='C2')
        self.c2.setVisible(False)
        self.view.addItem(self.c2)
        

        for marker_ind in range(self.num_markers):
            line = gl.GLLinePlotItem(pos=np.array([[0,0,0.],[1,1,1.]]),
                                     color=pg.glColor(colors[marker_ind]), width=2., antialias=True)
            line.hide()
            self.lines.append(line)
            self.view.addItem(line)
            
            pt = gl.GLScatterPlotItem(pos=np.array([[0,0,0.]]),
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
        self.nmarkers = gl.GLScatterPlotItem(pos=np.array([[0,0,0.]]),
                                             color=pg.glColor(colors[8]), size=5.)

        self.view.addItem(self.nmarkers)
        self.nmarkers.hide() #initially invisible

        # self.pbrot, is the rotation matrix to make the plumbline,
        self.pb = np.array([0,0,-1])
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
        if marked[0].size>0 and marked[1].size>0:
            valid_min = max([m.min() for m in marked])
            valid_max = min([m.max() for m in marked])
            valid_inds = np.arange(valid_min, valid_max+1, dtype='int')
        else:
            valid_inds = []

            
        if len(valid_inds)>0:
            # get undistorted points
            self.pts0 = self.ims[0].data[marker_ind, :2, valid_inds].T
            self.upts0 = cv.undistortPoints(self.pts0, self.ims[0].mtx, self.ims[0].dist,
                                            np.eye(3), self.ims[0].mtx)
        
            self.pts1 = self.ims[1].data[marker_ind, :2, valid_inds].T
            self.upts1 = cv.undistortPoints(self.pts1, self.ims[1].mtx, self.ims[1].dist,
                                            np.eye(3), self.ims[1].mtx)

            st_pts = cv.triangulatePoints(self.ims[0].proj, self.ims[1].proj,
                                          self.upts0, self.upts1)

            
            # normalize homogenous coordinates
            st_pts /= st_pts[-1]
            self.st = st_pts

            # swap y and z so z is up, y is dist from camera
            # and homogenous 4 to euclidian 3
            rmap = np.array([[1,0,0,0], 
                             [0,0,1,0], 
                             [0,-1,0,0]])

            # adjust downwards to plumbline
            st_pts = self.pbrot @ rmap @ st_pts

            self.st_pts = st_pts

            self.data[marker_ind, :, valid_inds] = st_pts.T


    def get_valid_inds(self, marker_ind):
        '''Find the indexes where between which we can interpolate a marker
        position

        '''
        marked = [np.where(im.data[marker_ind, -1])[0] for im in self.ims]
        if marked[0].size>0 and marked[1].size>0:
            valid_min = max([m.min() for m in marked])
            valid_max = min([m.max() for m in marked])
            valid_inds = np.arange(valid_min, valid_max+1, dtype='int')
        else:
            valid_inds = []

        return valid_inds

            
    def pan(self, marker_ind):
        '''Try to change the view focus

        '''
        # undo the old pan
        back = -self.focus
        self.view.pan(*back)
        self.focus[:] = 0,0,0

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
                x, y, z = self.data[marker_ind,:,valid_inds].mean(0)

            else:
                x, y, z = 0.,0.,0.

            self.view.pan(x,y,z)
            self.focus[:] = x,y,z
            

    def show_lines(self):
        '''Checks if we should draw each line (if it has some valid inds),
        then draws it based on existing, updated data, otherwise hides
        it.

        '''
        # print('show lines')
        xmax, ymax, zmax = 20,20,20
        for marker_ind in np.arange(self.num_markers):
            valid_inds = self.get_valid_inds(marker_ind)

            if len(valid_inds) > 0:
                line_data = self.data[marker_ind, :, valid_inds]
                self.lines[marker_ind].setData(pos=line_data)
                self.lines[marker_ind].show()

                xm, ym, zm = self.data[marker_ind, :, valid_inds].max(0)
                if xm>xmax: xmax = xm
                if ym>ymax: ymax = ym
                if zm>zmax: zmax = zm
                
            else:
                self.lines[marker_ind].hide()

        # does this frame have visible markers?
        if self.nmarkers.visible():
            xm, ym, zm = self.nmarkers.pos.max(0)
            if xm>xmax: xmax = xm
            if ym>ymax: ymax = ym
            if zm>zmax: zmax = zm
            
        # resizes the grid
        flr = 2*max(xmax, ymax)
        spc = 10**(np.floor(np.log10(flr)) -1)
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

        # if this is a calibration frame, showing in images, put
        # points in the 3d view too
        if n in self.ims[0].cal_inds and n in self.ims[1].cal_inds:
            cal_ind0 = self.ims[0].cal_inds.index(n)
            cal_ind1 = self.ims[1].cal_inds.index(n)
            
            h_pts = cv.triangulatePoints(self.ims[0].proj, self.ims[1].proj,
                                         self.ims[0].image_corners[cal_ind0][:,0].T,
                                         self.ims[1].image_corners[cal_ind1][:,0].T)
            # divide out for homogenous coordinates
            h_pts[:3] /= h_pts[3]
            
            rmap = np.array([[1,0,0,0], 
                             [0,0,1,0], 
                             [0,-1,0,0]])
            
            h_pts = rmap @ h_pts

            self.h_pts = h_pts

            
            self.nmarkers.setData(pos=h_pts.T)
            self.nmarkers.show()
        else:
            self.nmarkers.hide()

        self.show_lines()




    def set_camera_markers(self):
        '''Update the position of the cameras, after we have a T matrix. C1
        remains at the origin.

        '''
        # print('set camera markers')
        rr = -np.dot(self.R.T, self.T[:,0])
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
                self.cal_inds.append(ind)
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
                l_ids = left_ids[l_ind].flatten()
                r_ids = right_ids[r_ind].flatten()
                common_ids = np.intersect1d(l_ids, r_ids)

                if common_ids.size < 6:
                    print(f"{ind} only {common_ids.size} shared ChArUco corners")
                    continue

                l_take = np.nonzero(np.isin(l_ids, common_ids))[0]
                r_take = np.nonzero(np.isin(r_ids, common_ids))[0]

                # Sort both views by corner ID so object/image correspondences match.
                l_take = l_take[np.argsort(l_ids[l_take])]
                r_take = r_take[np.argsort(r_ids[r_take])]

                self.board_corners.append(self.ims[0].board_corners[l_ind][l_take])
                self.image0_corners.append(self.ims[0].image_corners[l_ind][l_take])
                self.image1_corners.append(self.ims[1].image_corners[r_ind][r_take])
            else:
                self.board_corners.append(self.ims[0].board_corners[l_ind])
                self.image0_corners.append(self.ims[0].image_corners[l_ind])
                self.image1_corners.append(self.ims[1].image_corners[r_ind])

        if len(self.board_corners) < 3:
            raise ValueError('Need at least 3 stereo calibration frames with shared board points')

            
        out = cv.stereoCalibrate(self.board_corners,
                                 self.image0_corners, self.image1_corners,
                                 self.ims[0].mtx, self.ims[0].dist,
                                 self.ims[1].mtx, self.ims[1].dist,
                                 (self.ims[0].w, self.ims[0].h),
                                 criteria = st_criteria, flags = st_flags)
        
        self.rmse, self.l_cm, self.l_dist, self.r_cm, self.r_dist, self.R, self.T, self.E, self.F = out
        

        # projection matrixes the product of the camera matrix and the rotation and translation matrixes 
        self.ims[0].proj = self.ims[0].mtx @ cv.hconcat([np.eye(3), np.zeros((3,1))])
        self.ims[1].proj = self.ims[1].mtx @ cv.hconcat([self.R, self.T])

        self.got_cal = True

        self.set_camera_markers()
        

    def get_plumbline(self):
        '''Use markers 1 and 2 to calculate a plumbline matrix.

        '''
        # find the frame indexes with markers 1 and 2, in both l and r views
        l1_inds = self.ims[0].data[0,-1]==1
        l2_inds = self.ims[0].data[1,-1]==1
        r1_inds = self.ims[1].data[0,-1]==1
        r2_inds = self.ims[1].data[1,-1]==1
        inds = np.where(l1_inds & l2_inds & r1_inds & r2_inds)[0]

        if len(inds)>0:

            m1 = self.data[0,:,inds].mean(0)
            m2 = self.data[1,:,inds].mean(0)

            # plumbline is the vector pointing toward the true down
            self.pb = m2 - m1

            # unit vector plumbline
            upb = self.pb/np.linalg.norm(self.pb)
            
            # target vector for down
            target = np.array([0, 0, -1])

            # axis of rotation: cross product of unit plumblin and target
            axis = np.cross(upb, target)

            # normalizing the axis to make it a unit vector
            axis = axis / np.linalg.norm(axis)

            # angle of rotation: arccosine of the dot product of unit
            # plumblin and target
            angle = np.arccos(np.dot(upb, target))

            # rodrigues' rotation formula to compute the rotation matrix
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            self.pbrot = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

            self.got_pb = True
            return 'found new plumbline'

        else:
            
            self.pbrot = np.eye(3)
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
        self.null_interp = InterpolatedUnivariateSpline([0,0.01], [np.nan,np.nan], k=1)
        self.interp = [[self.null_interp for xy in range(2)] for m in range(self.num_markers)]
        # markers for data
        self.markers = [pg.TargetItem() for marker in range(self.num_markers)]
        for marker_ind, marker in enumerate(self.markers):
            marker.setToolTip(str(marker_ind+1))
            marker.setPen(QtGui.QPen(QtGui.QColor(*colors[marker_ind])))
            # marker.sigPositionChanged.connect(self.marker_moved)
            marker.hide() #initially invisible
            self.imview.addItem(marker)

        # markers that aren't editable (for calibration)
        self.nmarkers = pg.ScatterPlotItem(symbol='o', setPen=(245,245,30,127))
        self.imview.addItem(self.nmarkers)
        self.nmarkers.clear() #initially invisible

        self.mousepos = (0,0)
        self.cal_inds = []

        # for communicating with the main window
        self.parent = parent

        
    def load_avi(self, fn):
        '''Get an avi file and read and display the first frame'''
        self.fn = fn
        self.cap = cv.VideoCapture(fn)
        self.num_frames  = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
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
            try:       # this will only work if we are in the domain (interpolating)
                x = self.interp[marker_ind][0](self.frame_ind)
                y = self.interp[marker_ind][1](self.frame_ind)
            except:    # if we are extrapolating, return nan, so we don't draw a marker
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
        previous = marker_ind, frame_ind, self.data[marker_ind,:,self.frame_ind].copy()
            
        if add:
            if pos is not None:
                x,y,s = pos
            else:
                x = self.imview.imageItem.mapFromScene(self.mousepos).x()
                y = self.imview.imageItem.mapFromScene(self.mousepos).y()
                s = 1
                
            self.data[marker_ind,:,self.frame_ind] = [x,y,s]

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
            x,y,marked = self.get_marker(marker_ind)

            # if we didn't get an xy, turn off the marker
            if np.isnan(x+y):
                self.markers[marker_ind].hide()

            else:
                alpha = 200 if marked else 50
                brush = pg.mkBrush(*colors[marker_ind], alpha)
                
                self.markers[marker_ind].setBrush(brush)
                self.markers[marker_ind].setPos((x,y))
                self.markers[marker_ind].show()

        # if this is a calibration frame
        if self.frame_ind in self.cal_inds:
            cal_ind = self.cal_inds.index(self.frame_ind)
            points = self.image_corners[cal_ind][:,0]
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
            try:       # this will only work if we are in the domain (interpolating)
                x = self.interp[marker_ind][0](frame_ind)
                y = self.interp[marker_ind][1](frame_ind)
            except:    # if we are extrapolating, return nan, so we don't draw a marker
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
        if num_filled_frames>=2:
            kval = np.clip(num_filled_frames -1, 1,3)

            # get values at marked frames
            ts = np.where(self.data[marker_ind, -1]==1)
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
            
        if pos.button==2 or (pos.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier) == QtCore.Qt.KeyboardModifier.AltModifier:
            add = False
        else:
            add = True
            
        self.set_marker(self.parent.curr_marker, add=add)

        
    def get_calibration(self, inds, num_rows=7, num_cols=6, side_len=11.5):
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

            corners_found, corners = cv.findChessboardCornersSB(
                roi, (num_rows, num_cols), chess_flags
            )

            if corners_found:
                corners = cv.cornerSubPix(
                    roi, corners, (11, 11), (-1, -1), criteria
                )
                corners += [x_min, y_min]
                self.image_corners.append(corners)
                self.board_corners.append(board_pts.copy())
                self.cal_inds.append(ind)
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
            legacy_pattern=False):
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
            #dictionary_id = cv.aruco.DICT_4X4_50
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

            (x_min, x_max), (y_min, y_max) = self.imview.getView().viewRange()
            x_min = max(int(np.floor(x_min)), 0)
            x_max = min(int(np.ceil(x_max)), gray.shape[1])
            y_min = max(int(np.floor(y_min)), 0)
            y_max = min(int(np.ceil(y_max)), gray.shape[0])

            roi = i[y_min:y_max, x_min:x_max]
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

            charuco_corners = charuco_corners.astype(np.float32)
            charuco_corners += [x_min, y_min]
            charuco_ids = charuco_ids.astype(np.int32)
            obj_pts = board_corners[charuco_ids.flatten()]

            self.image_corners.append(charuco_corners)
            self.board_corners.append(obj_pts)
            self.image_corner_ids.append(charuco_ids)
            self.cal_inds.append(ind)

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


    def get_circle_calibration(self, inds, num_rows=4, num_cols=11, side_len=10.0):
        self.cal_inds = []
        self.image_corners = []
        self.board_corners = []
        self.image_corner_ids = []

        # object points for asymmetric circles grid
        board_pts = np.zeros((num_rows * num_cols, 3), np.float32)
        k = 0
        for r in range(num_rows):
            for c in range(num_cols):
                board_pts[k, 0] = (2 * c + (r % 2)) * side_len
                board_pts[k, 1] = r * side_len
                board_pts[k, 2] = 0
                k += 1

        for ind in inds:
            self.cap.set(cv.CAP_PROP_POS_FRAMES, ind)
            im_found, im = self.cap.read()

            if not im_found or im is None:
                print(f"{ind} no im found")
                continue

            gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

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

            corners_found, centers = cv.findCirclesGrid(
                roi,
                (num_cols, num_rows),
                flags=cv.CALIB_CB_ASYMMETRIC_GRID
            )

            if corners_found:
                centers = centers.astype(np.float32)
                centers += [x_min, y_min]
                self.image_corners.append(centers)
                self.board_corners.append(board_pts.copy())
                self.cal_inds.append(ind)
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
        
        


####################
### stereography ###
####################
# sttt

class Stereography_window(QtWidgets.QMainWindow): #QWidget
    '''Window with two camera views, top, camera placement and 3d render,
    bottom. This is the main window for pyqtgraph.

    '''

    def __init__ (self, rot=False, cal_rows=8, cal_cols=11, cal_side=15):
        super(Stereography_window, self).__init__()
        self.window_title = 'Tracker 8'
        self.setWindowTitle(self.window_title)
        self.setMinimumWidth(900)
        self.setMinimumHeight(1000)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.cur = QtGui.QCursor

        self.num_cams = 2    #left and right
        self.num_markers = 9  #1-9
        self.rot = True
        self.fns = ['', '']
        self.chk_calibration = ''
        self.pb_calibration = ''
        self.cal_rows = cal_rows
        self.cal_cols = cal_cols
        self.cal_side = cal_side
        self.charuco_marker_fraction = 0.7
        self.charuco_dictionary = cv.aruco.DICT_5X5_100 if hasattr(cv, 'aruco') else None
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
        cal_menu = menubar.addMenu('Board Calibration')

        get_chk_action = QtWidgets.QAction('Compute Checkerboard', self)
        get_chk_action.triggered.connect(self.get_checkerboard)

        get_circ_action = QtWidgets.QAction('Compute Circle Grid', self)
        get_circ_action.triggered.connect(self.get_circle_grid)

        get_charuco_action = QtWidgets.QAction('Compute ChArUco Board', self)
        get_charuco_action.triggered.connect(self.get_charuco_board)

        insertTableMenu = QtWidgets.QMenu('Set Board Size', self)
        cal_menu.addMenu(insertTableMenu)

        selector = TableSizeSelector(st_win=self)
        widgetAction = QtWidgets.QWidgetAction(self)
        widgetAction.setDefaultWidget(selector)
        insertTableMenu.addAction(widgetAction)

        cal_menu.addActions([get_chk_action, get_circ_action, get_charuco_action])
        cal_menu.addSeparator()

        load_chk_action = QtWidgets.QAction('Load board calibration file...', self)
        load_chk_action.setShortcut(QtCore.Qt.SHIFT + QtCore.Qt.CTRL + QtCore.Qt.Key_C)
        load_chk_action.triggered.connect(self.load_checkerboard)
        cal_menu.addAction(load_chk_action)

        save_chk_action = QtWidgets.QAction('Save board calibration file...', self)
        save_chk_action.triggered.connect(self.save_checkerboard)
        cal_menu.addAction(save_chk_action)

        #  plumbline menu
        plumb_menu = menubar.addMenu('Plumbline')

        get_pb_action = QtWidgets.QAction('Compute Plumbline', self)
        get_pb_action.triggered.connect(self.get_plumbline)

        load_pb_action = QtWidgets.QAction('Load plumbline file...', self)
        load_pb_action.setShortcut(QtCore.Qt.SHIFT + QtCore.Qt.CTRL + QtCore.Qt.Key_P)
        load_pb_action.triggered.connect(self.load_plumbline)

        save_pb_action = QtWidgets.QAction('Save plumbline file...', self)
        save_pb_action.triggered.connect(self.save_plumbline)

        plumb_menu.addActions([get_pb_action])
        plumb_menu.addSeparator()
        plumb_menu.addActions([load_pb_action, save_pb_action])


        #  markers menu
        mark_menu = menubar.addMenu('Markers')
        marker_group = QtWidgets.QActionGroup(self)

        actions = []
        for n in range(9):
            action = QtWidgets.QWidgetAction(self, checkable=True)
            radio = QtWidgets.QRadioButton(f'Marker {n+1}', self)
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
        namespace = {'pg': pg, 'np': np, 'st':self}
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
        self.chk_indicator = QtWidgets.QPushButton(text='checkerboard', parent=self)
        self.chk_indicator.clicked.connect(self.load_checkerboard)
        self.chk_indicator.setMinimumHeight(25)
        self.chk_indicator.setMaximumHeight(30)
        self.set_calibration_indicator('chk', 'None')
        # plumbline calibration button
        self.pb_indicator = QtWidgets.QPushButton(text='plumbline', parent=self)
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
        self.num_frames = 1   #until we load a dir
        self.marker_keys = [QtCore.Qt.Key.Key_1, QtCore.Qt.Key.Key_2, QtCore.Qt.Key.Key_3, QtCore.Qt.Key.Key_4, QtCore.Qt.Key.Key_5, QtCore.Qt.Key.Key_6, QtCore.Qt.Key.Key_7, QtCore.Qt.Key.Key_8, QtCore.Qt.Key.Key_9]

        # print the help text
        self.print_help()
        
        ### double control action flags
        self.ctrl_x = False
        self.ctrl_c = False
        
        ### keypress actions
        self.key_actions = {
            # prefixes
            ('control C')           : (self.set_ctrl_action, ('C',)),
            ('control X')           : (self.set_ctrl_action, ('X',)),
            # undo
            ('control Z')           : (self.undo, ()),
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
            ('`')                   : (self.reset_view, ()),
            # add markers
            ('1')                   : (self.move_marker,       (None,)),
            ('2')                   : (self.move_marker,       (None,)),
            ('3')                   : (self.move_marker,       (None,)),
            ('4')                   : (self.move_marker,       (None,)),
            ('5')                   : (self.move_marker,       (None,)),
            ('6')                   : (self.move_marker,       (None,)),
            ('7')                   : (self.move_marker,       (None,)),
            ('8')                   : (self.move_marker,       (None,)),
            ('9')                   : (self.move_marker,       (None,)),
            # remove markers
            ('alt 1')               : (self.move_marker,       (None, False)),
            ('alt 2')               : (self.move_marker,       (None, False)),
            ('alt 3')               : (self.move_marker,       (None, False)),
            ('alt 4')               : (self.move_marker,       (None, False)),
            ('alt 5')               : (self.move_marker,       (None, False)),
            ('alt 6')               : (self.move_marker,       (None, False)),
            ('alt 7')               : (self.move_marker,       (None, False)),
            ('alt 8')               : (self.move_marker,       (None, False)),
            ('alt 9')               : (self.move_marker,       (None, False)),
            # fullscreen toggle
            ('control F')           : (self.toggle_fullscreen, ())
        }

        
        #fresh undo list
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
        d,f = os.path.split(fn)

        # look for the corresponding file, starting with L or R
        if f.startswith('L') and os.path.isfile(os.path.join(d,f'R{f[1:]}')):
            self.fns = [os.path.join(d,f), os.path.join(d,f'R{f[1:]}')]
            
        elif f.startswith('R') and os.path.isfile(os.path.join(d,f'L{f[1:]}')):
            self.fns = [os.path.join(d,f'L{f[1:]}'), os.path.join(d,f)]

        # or if fn dialog was cancelled
        elif f=='':
            return

        # or open a dialog to get the other file
        else: 
            # self.fns = [None, None]
            fn = str(QtWidgets.QFileDialog.getOpenFileName(self, 'Select other video', '', 'avi files (*.avi)')[0])
            d2,f2 = os.path.split(fn)

            # if dialog was cancelled
            if f2 == '':
                return
            # otherwise this is the second
            else:
                self.fns = [os.path.join(d,f), os.path.join(d2,f2)]


        # load the images
        for im, avi in zip(self.ims, self.fns):
            im.load_avi(avi)

        # how many frames?
        self.num_frames  = self.ims[0].num_frames
        self.frame_slider.setMaximum(self.num_frames-1)
        self.frame_slider.setValue(0)
        self.change_frame(0)

        # set blank data
        data = [None]*(len(self.ims)+1)
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


    def get_checkerboard(self):
        '''Search each imframe for calibration (checkerboard right now)
        images, and construct camera matrixes. Then, use them together
        in the tdframe to construct projection matrixes.

        '''
        # get the inds where any image is marked
        inds = np.where(np.logical_or(*[im.data[0,-1] for im in self.ims]))[0]
        print(f'{inds=}')
        # if there are none, just mark num of them, evenly
        if len(inds) == 0:
            num = 20 
            inds = np.linspace(0, self.num_frames-1, num, dtype='int')
            print('no inds marked')

        # calibrate images and three D
        print(f'{inds=}\n')
        ok = []
        for im in self.ims:
            ok.append(
                im.get_calibration(
                    inds,
                    num_rows=self.cal_rows,
                    num_cols=self.cal_cols,
                    side_len=self.cal_side
                )
            )

        if not all(ok):
            self.console_write("Checkerboard calibration failed in one or both cameras.", "calibration")
            return

        self.td.get_calibration()
        
        
        # for im in self.ims:
        #     im.get_calibration(inds, num_rows=self.cal_rows, num_cols=self.cal_cols, side_len=self.cal_side)
        # self.td.get_calibration()


        
        self.console_write(f'frames: {self.td.cal_inds}\nroot mean squared error = {self.td.rmse}', 'calibration')

    def get_circle_grid(self):
        # undo the checkerboard subtraction
        rows = self.cal_rows + 1
        cols = self.cal_cols + 1

        # same frame-selection logic as checkerboard
        inds = np.where(np.logical_or(*[im.data[0, -1] for im in self.ims]))[0]
        print(f'{inds=}')
        if len(inds) == 0:
            num = 20
            inds = np.linspace(0, self.num_frames - 1, num, dtype='int')
            print('no inds marked')

        print(f'{inds=}\n')

        ok = []
        for im in self.ims:
            ok.append(
                im.get_circle_calibration(
                    inds,
                    num_rows=rows,
                    num_cols=cols,
                    side_len=self.cal_side
                )
            )

        if not all(ok):
            self.console_write("Circle-grid calibration failed.", "calibration")
            return

        self.td.get_calibration()
        self.console_write(
            f'frames: {self.td.cal_inds}\nroot mean squared error = {self.td.rmse}',
            'calibration'
        )

        



    def get_charuco_board(self):
        """Compute a stereo calibration from ChArUco board frames."""
        # ChArUcoBoard wants the number of squares. The checkerboard method
        # stores inner-corner counts, so add one in each dimension.
        rows = self.cal_rows + 1
        cols = self.cal_cols + 1
        marker_len = self.cal_side * self.charuco_marker_fraction

        inds = np.where(np.logical_or(*[im.data[0, -1] for im in self.ims]))[0]
        print(f'{inds=}')
        if len(inds) == 0:
            num = 20
            inds = np.linspace(0, self.num_frames - 1, num, dtype='int')
            print('no inds marked')

        print(f'{inds=}\n')

        ok = []
        for im in self.ims:
            ok.append(
                im.get_charuco_calibration(
                    inds,
                    num_rows=rows,
                    num_cols=cols,
                    side_len=self.cal_side,
                    marker_len=self.charuco_marker_len,
                    dictionary_id=self.charuco_dictionary
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
        self.td.get_plumbline()

        self.console_write(f'{self.td.pbrot}', 'plumbline rotation')
        

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
                x,y,z = self.td.data[marker_ind,:,frame_ind]
                r,g,b = colors[marker_ind]
                t = f"<pre style='color: rgb({r}, {g}, {b});'>{marker_ind+1}: {x:7.01f} {y:7.01f} {z:7.01f}</pre>"
                text.append(t)
                # self.t = t

        self.tab.setText('\n'.join(text))

        
    def change_frame(self, frame_ind=None, autorange=False, autolevel=False):
        '''Changes to a new frame in the avi by updating the images, 3d view,
        and the table of marker position 

        '''
        if frame_ind is None:
            frame_ind = self.frame_slider.value()
        self.frame_value.setText('{}/{}'.format(self.frame_slider.value(), self.num_frames-1))

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


    def _marked_frames(self):
        """Return sorted unique frame indices marked in any image."""
        marked_frames = []

        for im in self.ims:
            where_marked = np.where(im.data[:, -1] == 1)

            if len(where_marked) > 1:
                marked_frames.append(where_marked[1])
            else:
                marked_frames.append(where_marked[0])

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
            marked = self._marked_frames()
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
        if key=='X':
            self.ctrl_x = True
        elif key=='C':
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
            
                
    def move_marker(self, event=None, add=True, marker_ind=None):
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

            data_fn = str(QtWidgets.QFileDialog.getSaveFileName(self, 'Save Data As', ddefaultname, 'npy files (*.npy)')[0])
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

            npy_fn = str(QtWidgets.QFileDialog.getSaveFileName(self, 'Save Data As', ddefaultname, 'npy files (*.npy)')[0])
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

            csv_fn = str(QtWidgets.QFileDialog.getSaveFileName(self, 'Save Data As', ddefaultname, 'csv files (*.csv)')[0])
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

        if fn=='':
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
            self.console_write('Need checkerboard first')
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

        fn = str(QtWidgets.QFileDialog.getSaveFileName(self, 'Save Checkerboard', cdefaultname, 'npy files (*.npy)')[0])
        
        data = np.zeros((27,5))
        
        # camera matrixes
        data[ 0:3 , 0:3] = self.ims[0].mtx
        data[ 3:6 , 0:3] = self.ims[1].mtx
        data[ 6:9 , 0:3] = self.ims[0].nmtx
        data[ 9:12, 0:3] = self.ims[1].nmtx
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

        self.console_write(fn, 'saved checkerboard')
        
        
        
    def save_plumbline(self, fn=None):
        '''Save the plumbline rotation matrix, which points the scene down.

        '''
        if self.ims[0].fn == '' or self.td.got_pb is None:
            self.console_write('Need plumbline first')
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

        fn = str(QtWidgets.QFileDialog.getSaveFileName(self, 'Save Plumbline Calibration', pdefaultname, 'npy files (*.npy)')[0])
        
        np.save(fn, self.td.pbrot)

        self.set_calibration_indicator('pb', fn=fn)
        
        self.console_write(fn, 'saved plumbline')
        
            
    def load_checkerboard(self, fn=None):
        '''Get the checkerboard calibration.

        '''
        fn = str(QtWidgets.QFileDialog.getOpenFileName(self, 'Select checkerboard calibration', '', 'npy files (*.npy)')[0])

        if fn=='':
            return

        try:
            data = np.load(fn)
            assert data.shape == (27,5)

        except Exception as e:
            self.console_write(f"An error occurred: {e}")
            return
            
        # camera matrixes
        self.ims[0].mtx  = data[ 0:3 , 0:3]
        self.ims[1].mtx  = data[ 3:6 , 0:3]
        self.ims[0].nmtx = data[ 6:9 , 0:3]
        self.ims[1].nmtx = data[ 9:12, 0:3]
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

        self.console_write(fn, f'loaded checkerboard')

    
    def load_plumbline(self, fn=None):
        '''Get the plumbline calibration.

        '''
        fn = str(QtWidgets.QFileDialog.getOpenFileName(self, 'Select plumbline file', '', 'npy files (*.npy)')[0])

        if fn=='':
            return

        try:
            data = np.load(fn)
            assert data.shape == (3,3)

        except Exception as e:
            self.console_write(f"An error occurred: {e}")
            return

        self.td.pbrot = data
        self.td.got_pb = True

        self.set_calibration_indicator('pb', fn=fn)

        self.console_write(fn, f'loaded plumbline')

        
        
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
        if fn=='None':
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
            out = f'{subject.capitalize()}:\n{"="*(len(subject)+1)}\n{out}'
        self.console.write(out)

            
    def undo(self):
        '''Pops the command off the end of the undo list, then executes
        it. The command includes the imframe it came from and the
        marker state. Because changing the marker state adds the
        previous state to the undo list, we pop that off, so iterative
        undo goes back down the list, not back and forth between the
        two most recent states

        '''
        if len(self.undo_list)>0:
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

