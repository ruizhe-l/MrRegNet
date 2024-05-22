import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, MultipleLocator, AutoLocator
from matplotlib.colors import BoundaryNorm

import cv2
import SimpleITK as sitk

def plot_grid(x,y, ax=None, **kwargs):
    ax = ax or plt.gca()
    segs1 = np.stack((x,y), axis=2)
    segs2 = segs1.transpose(1,0,2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()

def draw_grid(dpi=100, size=(256,256)):

    _y, _x = np.mgrid[:20:1, :20:1]
    fig = Figure(figsize=(6,6), dpi=dpi)
    fig.patch.set_facecolor('black')
    fig.add_axes([0, 0, 1, 1])
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    plot_grid(_x, _y, ax=ax, color="white")
    # plot_grid(grid_x, -grid_y, ax=ax, color="red")
    
    ax.axis('off')
    canvas.draw()
    np_frame = np.asarray(canvas.buffer_rgba(), dtype='uint8')[...,0]
    np_frame = cv2.resize(np_frame, size, interpolation=cv2.INTER_NEAREST)
    np_frame[np_frame < 128] = 0
    np_frame[np_frame >= 128] = 255

    return np_frame

def _vis_grid(disp, nvec=16):  
    u = -disp[..., 1]
    v = -disp[..., 0]

    nl, nc, _ = disp.shape
    step = max(1, max(nl//nvec, nc//nvec))

    y, x = np.mgrid[:nl:step, :nc:step]

    u_ = u[::step, ::step]
    v_ = v[::step, ::step]
    # grid_x = x + u_ * shape[1] / disp.shape[1]
    # grid_y = y + v_ * shape[0] / disp.shape[0]
    grid_x = x + u_
    grid_y = y + v_

    fig = Figure(figsize=(5,5))
    ax =fig.patch.set_facecolor('white')
    
    canvas = FigureCanvas(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    # x_border = [[x.min(), x.max()], [x.min(), x.max()]]
    # y_border = [[y.min(), y.min()], [y.max(), y.max()]]
    # plot_grid(x, y, ax=ax, color="red")
    plot_grid(grid_x, grid_y, ax=ax, color="black")
    
    pad_x = 0.1 * (x.max() - x.min())
    pad_y = 0.1 * (y.max() - y.min())
    ax.set_xlim(0-pad_x, x.max()+pad_x)  # Set x-axis limits without padding
    ax.set_ylim(y.max()+pad_y, 0-pad_y)  # Set y-axis limits without padding
    ax.axis('off')
    # ax.margins(0)
    canvas.draw()
    np_frame = np.asarray(canvas.buffer_rgba(), dtype='uint8')[...,:3]
    return np_frame


def _vis_quiver(disp, nvec=32, vmin=None, vmax=None, quiver=True, dsize=[256,256]): 
    
    #TODO: add color bar

    u = -disp[..., 1] * (dsize[1] / disp.shape[1])
    v = -disp[..., 0] * (dsize[0] / disp.shape[0])

    norm = np.sqrt(u ** 2 + v ** 2)

    nvec = nvec  # Number of vectors to be displayed along each image dimension
    nl, nc, _ = disp.shape
    step = max(1, max(nl//nvec, nc//nvec))

    y, x = np.mgrid[:nl:step, :nc:step]

    u_ = u[::step, ::step]
    v_ = v[::step, ::step]

    fig = Figure(figsize=(5,5))
    fig.add_axes([0, 0, 0.9, 1])
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    vis1 = ax.imshow(norm, vmin=vmin, vmax=vmax, cmap='jet')
    if quiver:
        ax.quiver(x, y, u_, v_, color='r', units='dots',
        angles='xy', scale_units=None, lw=3)
    
    ax.axis('off')
    ax.margins(0)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.add_axes(cax)
    cbar = fig.colorbar(vis1, cax=cax)
    cbar.ax.tick_params(labelsize=18)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # divider = make_axes_locatable(ax)
    # cax = divider.new_vertical(size="5%", pad=0.05, pack_start=True)
    # fig.add_axes(cax)
    # fig.colorbar(vis1, cax=cax, orientation="horizontal")
        
    canvas.draw()
    np_frame = np.asarray(canvas.buffer_rgba(), dtype='uint8')[...,:3]
    return np_frame

def vis_disps(disps, shape, nvec=32, channel_first=True, vmin=None, vmax=None, show_quiver=False):
    disps = np.array(disps)
    if channel_first:
        disps = np.transpose(disps, [0,2,3,1])
    
    # regular_grid = get_regular_grid(disp.shape[1:-1])
    vmin = (None, ) * disps.shape[0] if vmin is None else vmin
    vmax = (None, ) * disps.shape[0] if vmax is None else vmax
    vis_results_quiver = []
    vis_results_grid = []
    for i, disp_ in enumerate(disps):
        disp_n = disp_.copy()
        # disp_n[..., 0] = disp_n[..., 0]*2/disp_n.shape[1]
        # disp_n[..., 1] = disp_n[..., 1]*2/disp_n.shape[0]
        vis_q = _vis_quiver(disp_n, nvec=nvec, vmin=vmin[i], vmax=vmax[i], quiver=show_quiver, dsize=shape)
        vis_g = _vis_grid(disp_n, nvec=nvec)
        vis_q = cv2.resize(vis_q, shape)
        vis_g = cv2.resize(vis_g, shape)
        if channel_first:
            vis_q = vis_q.transpose([2,0,1])
            vis_g = vis_g.transpose([2,0,1])
        vis_results_quiver.append(vis_q)
        vis_results_grid.append(vis_g)

    return [np.array(vis_results_quiver), np.array(vis_results_grid)]

def vis_heatmap_uneven(disp, nvec=32, vmin=None, vmax=None, quiver=True, dsize=[256,256]):
    u = -disp[..., 1] * (dsize[1] / disp.shape[1])
    v = -disp[..., 0] * (dsize[0] / disp.shape[0])

    img = np.sqrt(u ** 2 + v ** 2)
    cmap = plt.get_cmap('jet')
    ticks = [0,3,6,9,15,21] 
    bounds = np.array(list(np.linspace(0, 9, 100)) + list(np.linspace(9, 21, 40)[1:]))
    norm = BoundaryNorm(bounds, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig = Figure(figsize=(5,5))
    fig.add_axes([0, 0, 0.9, 1])
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    vis1 = ax.imshow(img, cmap=cmap, norm=norm)

    
    ax.axis('off')
    ax.margins(0)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.add_axes(cax)
    cbar = plt.colorbar(sm, cax=cax, ticks=ticks)
    # cbar.ax.tick_params(labelsize=18)
    # cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # divider = make_axes_locatable(ax)
    # cax = divider.new_vertical(size="5%", pad=0.05, pack_start=True)
    # fig.add_axes(cax)
    # fig.colorbar(vis1, cax=cax, orientation="horizontal")
        
    canvas.draw()
    np_frame = np.asarray(canvas.buffer_rgba(), dtype='uint8')[...,:3]
    return np_frame


def vis_heatmap(disps, shape, nvec=32, channel_first=True, vmin=None, vmax=None):
    disps = np.array(disps)
    if channel_first:
        disps = np.transpose(disps, [0,2,3,1])
    
    # regular_grid = get_regular_grid(disp.shape[1:-1])
    vis_results_quiver = []
    for disp_ in disps:
        disp_n = disp_.copy()
        # disp_n[..., 0] = disp_n[..., 0]*2/disp_n.shape[1]
        # disp_n[..., 1] = disp_n[..., 1]*2/disp_n.shape[0]
        vis_q = _vis_quiver(disp_n, nvec=nvec, vmin=vmin, vmax=vmax, quiver=False)
        vis_q = cv2.resize(vis_q, shape)
        if channel_first:
            vis_q = vis_q.transpose([2,0,1])
        vis_results_quiver.append(vis_q)

    return np.array(vis_results_quiver)

def generate_grid(imgshape):
    if len(imgshape) == 3:
        grid = np.mgrid[:imgshape[2], :imgshape[1], :imgshape[0]].transpose((1,2,3,0))
        grid = np.swapaxes(grid,0,2)
    else:
        grid = np.mgrid[:imgshape[1], :imgshape[0], :imgshape[0]].transpose((1,2,0))
        grid = np.swapaxes(grid,0,1)
    return grid


def JacboianDetSitk(disps, channel_first=True):
    if channel_first:
        disps = np.transpose(disps, [0,*range(2,len(disps.shape)),1])
    jacs = []
    for i in range(disps.shape[0]):
        dispi = sitk.GetImageFromArray(disps[i], isVector=True)
        jacof = sitk.DisplacementFieldJacobianDeterminantFilter()
        jaci = jacof.Execute(dispi)
        jac = sitk.GetArrayFromImage(jaci)
        jacs.append(jac)
    return np.array(jacs)

# def JacboianDet(disps, channel_first=True):
#     if channel_first:
#         disps = np.transpose(disps, [0,*range(2,len(disps)),1])

#     if len(disps.shape) == 4:    
#         J = disps + np.expand_dims(generate_grid(disps.shape[:2]), 0)
#         dx = J[:, 1:, :-1, :] - J[:, :-1, :-1, :]
#         dy = J[:, :-1, 1:, :] - J[:, :-1, :-1, :]
#     else:
#         J = disps + np.expand_dims(generate_grid(disps.shape[1:4]), 0)
#         dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
#         dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
#         dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

#         Jdet0 = dx[:,:,:,:,0] * (dy[:,:,:,:,1] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,1])
#         Jdet1 = dx[:,:,:,:,1] * (dy[:,:,:,:,0] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,0])
#         Jdet2 = dx[:,:,:,:,2] * (dy[:,:,:,:,0] * dz[:,:,:,:,1] - dy[:,:,:,:,1] * dz[:,:,:,:,0])
#         Jdet = Jdet0 - Jdet1 + Jdet2

#     return np.array(Jdet)

