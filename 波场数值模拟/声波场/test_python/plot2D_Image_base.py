import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from random import uniform 


def main():
  #goPlotVel()
  #goPlotSnap()
  goPlotVel()

def goPlotVel():
  path = '/home/diskc/ryjin/fd2d_pml/vel/'
  vfile = 'garben_p_0401x0301.bin' 
  image_shape = [401,301]
  vel = Read2DImage(path+vfile,image_shape)
  vel = vel/1000.
  plotcolorbar = 1
  print(vel.min(),vel.max())
  norm = mpl.colors.Normalize(2,4.5)
  plot2DImage('../figure/',vfile.rstrip('.bin')+'_cbar',vel.T,
              cmap='rainbow',datatype='velocity',
              dx=0.01,dz_or_dt=0.01,norm=norm,plotcolorbar=plotcolorbar)
  plt.show()

def goPlotSnap():
  path = '/home/diskc/ryjin/fd2d_pml/bin/'
  sxfile = 'snapshot_0200.bin' 
  image_shape = [401,301]
  snap = Read2DImage(path+sxfile,image_shape)
  plotcolorbar = 1
  print(snap.min(),snap.max())
  snap /= snap.max()
  norm = mpl.colors.Normalize(-0.5,0.5)
  plot2DImage('../figure/',sxfile.rstrip('.bin')+'_cbar',snap.T,
              cmap='seismic',datatype='snapshot',
              dx=0.01,dz_or_dt=0.01,norm=norm,plotcolorbar=plotcolorbar)
  plt.show()

def goPlotVel():
  path = '/home/diskc/ryjin/fd2d_pml/data/'
  sgfile = 'data_0200.bin' 
  image_shape = [401,4001]
  sg = Read2DImage(path+sgfile,image_shape)
  plotcolorbar = 1
  print(sg.min(),sg.max())
  sg /= sg.max()
  norm = mpl.colors.Normalize(-0.01,0.01)
  plot2DImage('../figure/',sgfile.rstrip('.bin'),sg.T,
              cmap='gray',datatype='shot_gather',aspect=True,
              dx=0.01,dz_or_dt=0.001,norm=norm)
  plt.show()

def Read3DImage(fname,image_shape):
    image = np.fromfile(fname,dtype=np.single)
    image = np.reshape(image,image_shape)
    return image

def ZeroNorm(image):
    image_max = image.max()
    image_min = image.min()
    image1 = (image-image_min)/(image_max-image_min)
    return image1

def Read2DImage(filename,image_shape):
    image = np.fromfile(filename,dtype=np.single)
    image = np.reshape(image,image_shape)
    #image = image.transpose()
    return image

def add_right_cax(ax, pad, width):
  '''
  在一个ax右边追加与之等高的cax.
  pad是cax与ax的间距.
  width是cax的宽度.
  '''
  axpos = ax.get_position()
  caxpos = mpl.transforms.Bbox.from_extents(
       axpos.x1 + pad,
       axpos.y0,
       axpos.x1 + pad + width,
       axpos.y1
  )
  cax = ax.figure.add_axes(caxpos)
  return cax

def add_bottom_cax(ax, pad, hight):
  '''
  在一个ax下边追加与之等宽的cax.
  pad是cax与ax的间距.
  hight是cax的高度.
  '''
  axpos = ax.get_position()
  caxpos = mpl.transforms.Bbox.from_extents(
       axpos.x0,
       axpos.y0 - pad - hight,
       axpos.x1,
       axpos.y0 - pad
  )
  cax = ax.figure.add_axes(caxpos)
  return cax

def safe_locator(span, step_guess, max_ticks=1000):
    """生成安全的MultipleLocator，避免tick数量过多"""
    n_ticks = span / step_guess
    if n_ticks > max_ticks:
        step_guess = span / max_ticks * 1.1
    return mpl.ticker.MultipleLocator(step_guess)


def plot2DImage(path,filename,image,
                cmap=None,norm=None,aspect=None,
                colorbar=None,plotcolorbar=None,
                datatype='velocity',dx=10.0,dz_or_dt=1.0):
    """
    datatype:
        velocity  → Distance(km)–Depth(km)
        snapshot  → Distance(km)–Depth(km)
        shot_gather→ Distance(km)–Time(s)

    dx:     水平采样间隔（km）
    dz_or_dt: 垂向采样（km 或 s）
    """
    font = {'family' : 'Arial',
            'weight' : 'normal',
            'size'   : 15,
            }
    font2 = {'family' : 'Arial',
            'weight' : 'normal',
            'size'   : 12,
            }
    
    fig, ax = plt.subplots()
    nx, nz = image.shape[1], image.shape[0]  # 横纵尺寸
    
    # 横轴
    xspan = (nx-1) * dx
    xmajor = safe_locator(xspan, 0.8)
    xminor = safe_locator(xspan, 0.2)
    ax.set_xlabel('Distance (km)', font)

    # 纵轴根据 datatype
    yspan = (nz-1) * dz_or_dt

    if datatype in ['velocity', 'snapshot']:
        ymajor = safe_locator(yspan, 0.5)
        yminor = safe_locator(yspan, 0.1)
        ax.set_ylabel('Depth (km)', font)
    elif datatype == 'shot_gather':
        ymajor = safe_locator(yspan, 1)
        yminor = safe_locator(yspan, 0.2)
        ax.set_ylabel('Time (s)', font)
    else:
        raise ValueError("datatype must be 'velocity', 'snapshot', or 'shot_gather'")

    ax.xaxis.set_major_locator(xmajor)
    ax.xaxis.set_minor_locator(xminor)
    ax.yaxis.set_major_locator(ymajor)
    ax.yaxis.set_minor_locator(yminor)
    ax.tick_params(which='major', labelsize=12, width=1.5)
    ax.tick_params(which='minor', labelsize=12, width=1.0)
    
    # extent 参数保证坐标轴单位对齐
    extent = [0, xspan, yspan, 0]
    
    im = ax.imshow(image, cmap=cmap, norm=norm, aspect=aspect, extent=extent, origin='upper')
    
    #if invert_y:
    #    ax.invert_yaxis()  # Depth 向下增加

    # Colorbar
    if plotcolorbar:
        cax = add_right_cax(ax, pad=0.02, width=0.02)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=15)
        if datatype=='velocity':
            cbar.set_label('Velocity (km/s)', fontsize=15)
        elif datatype=='snapshot':
            cbar.set_label('Amplitude', fontsize=15)
        elif datatype=='shot_gather':
            cbar.set_label('Amplitude', fontsize=15)

    # ----------- 保存 -----------
    fig.savefig(f"{path}{filename}.png", dpi=200, bbox_inches='tight', pad_inches=0.0)
    fig.savefig(f"{path}{filename}.pdf", bbox_inches='tight', pad_inches=0.0)

    print(f"Saved {path}{filename}.png & .pdf")

if __name__ == '__main__':
    main()
