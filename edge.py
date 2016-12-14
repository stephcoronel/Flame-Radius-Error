import numpy as np
import time,sys,os
import cmath
import warnings
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.ticker import LogLocator
from IPython.core.pylabtools import figsize
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from math import factorial
from operator import itemgetter, attrgetter
import matplotlib.lines as mlines
import cantera as ct
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy.interpolate import griddata
from scipy.sparse import diags
from scipy.integrate import ode
import Image, ImageDraw
import cv2
import skimage
from ellipse import * # IMPORT FUNCTIONS FROM ellipse.py
#-------------------------------------------------------------------------------
# THE TWO LINES BELOW DEFINE THE FONT USED FOR THE FIGURES
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('text', usetex=True)
#-------------------------------------------------------------------------------
# IGNORE WARNINGS
np.seterr(divide='ignore', invalid='ignore')
#-------------------------------------------------------------------------------
# DEFINE THE SIZE OF THE FIGURES AND FONT SIZE OF FIGURE LABELS
figx = 5
figy = 3
lsize = 12 
#-------------------------------------------------------------------------------
# pathDIR IS THE DIRECTORY WHERE FIGURES WILL BE SAVED TO
pathDIR = 'Output/'
# CREATES THE pathDIR DIRECTORY IF IT ALREADY DOESN'T EXIST
try: 
    os.makedirs(pathDIR)
except OSError:
    if not os.path.isdir(pathDIR):
        raise
#-------------------------------------------------------------------------------
# DEFINE THE NAME THAT THE FIGURES WILL BE SAVED US
name = 'base'
# FLAME RADIUS DIAMETER
r = 10 # mm
# CONVERSION RATIO OF PIXELS TO MM
dxdp = 2*10.0/160.0
#-------------------------------------------------------------------------------
# MAKE A CIRCLE WITH DIAMETER OF 160 PX WHICH IS EQUAL TO r AND SAVE TO test.png
image = Image.new('RGB', (200, 200), 'black')
draw = ImageDraw.Draw(image)
draw.ellipse((20,20,180,180), fill = 'black', outline ='white')
image.save(pathDIR + 'test.png')
#-------------------------------------------------------------------------------
# LOAD test.png AND APPLY CANNY FILTER TO DETECT CIRCLE EDGE
img = cv2.imread(pathDIR + 'test.png',0)
edges = cv2.Canny(img,100,200)
#-------------------------------------------------------------------------------
# MAKE PLOT OR ORIGINAL CIRCLE AND DETECTED CANNY EDGE, SIDE BY SIDE
fig = plt.figure()
ax1 = plt.subplot2grid((1,2),(0,0)) # DEFINE 1x2 SUBPLOT LOCATED AT (0,0)
ax1.imshow(img,cmap='gray') # PLOTS THE IMAGE
ax1.set_title('Original Image') # SETS THE TITLE
ax1.set_xticks([]) # REMOVES TICK MARKS
ax1.set_yticks([]) # REMOVES TICK MARKS
ax2 = plt.subplot2grid((1,2),(0,1)) # DEFINE 1X2 SUPLOT LOCATED AT (0,1)
ax2.imshow(edges,cmap = 'gray') # PLOTS THE IMAGE 
ax2.set_title('Edge Image') # SETS THE TITLE
ax2.set_xticks([]) # REMOVES TICK MARKS
ax2.set_yticks([]) # REMOVES TICK MARKS
fig.set_size_inches(figx,figy) # SETS THE FIGURE SIZE
fig.savefig(pathDIR + name + '.pdf',dpi=100,bbox_inches='tight') # SAVE FIGURE
#-------------------------------------------------------------------------------
# EXTRACT CANNY DETECTED POINTS
xmat = np.nonzero(edges)[0]
ymat = np.nonzero(edges)[1]
#-------------------------------------------------------------------------------
# CONVERT TO SI UNITS
xmat = xmat*dxdp
ymat = ymat*dxdp
#-------------------------------------------------------------------------------
# CENTER CIRCLE AT (0,0)
xmat = (xmat - (np.max(xmat)+np.min(xmat))/2.0)
ymat = (ymat - (np.max(ymat)+np.min(ymat))/2.0)
#-------------------------------------------------------------------------------
# PLOT DETECTED CANNY EDGE
fig = plt.figure()
ax1 = plt.subplot2grid((1,1),(0,0))
ax1.plot(xmat,ymat,marker='.',ls='none',mew=1.0,mfc='none',mec='b',ms=1,label='original') # ls=linestyle, mew=marker edge width, mfc=marker face color, mec=marker edge width, ms=marker size
ax1.set_xlabel('$x$ (mm)',fontsize=lsize) # SET X LABEL
ax1.set_ylabel('$y$ (mm)',fontsize=lsize) # SET Y LABEL
ax1.set_aspect('equal') # SET 1:1 X AND Y ASPECT RATIO
ax1.grid(b=True, which='major', color='k', linestyle=':') # MAKE A GRID
fig.set_size_inches(figx,figy)
fig.savefig(pathDIR + name + '_xy.pdf',dpi=100,bbox_inches='tight')

#noise = 2
#nr = np.random.normal(0,noise/100.0,len(xmat))
#xmat = xmat + nr*xmat
#ymat = ymat + nr*ymat

#ax1.plot(xmat,ymat,marker='.',ls='none',mew=1.0,mfc='none',mec='k',ms=1,label='Gaussian noise')
#-------------------------------------------------------------------------------
# RUN ELLIPSE FUNCTIONS
a = fitEllipse(xmat,ymat)
axes = ellipse_axis_length(a)
a,b = axes # SEMI-MAJOR AND SEMI-MINOR AXES
print 'axes = ' + str(axes)
#-------------------------------------------------------------------------------
# CALCULATE THE SURFACE AREA OF AN OBLATE OR PROLATE ELLIPSOID
if a<b:
    e = np.sqrt(1-a**2/b**2)
    ellipArea = 2.0*np.pi*b**2*(1.0+(1.0-e**2)/e*np.arctanh(e))
elif a>b:
    e = np.sqrt(1-b**2/a**2)
    ellipArea = 2.0*np.pi*b**2*(1.0+a/(b*e)*np.arcsin(e))
#-------------------------------------------------------------------------------
# CALCULATE EQUIVALENT RADIUS BASED ON SURFACE AREA OF ELLIPSE = SURFACE AREA OF SPHERE
requiv = np.sqrt(ellipArea/(4.0*np.pi))
#-------------------------------------------------------------------------------
# PLOT ELLIPSE BASED ON a and b
xfit = np.arange(-10,10,0.0001)
yfithigh = np.sqrt(b**2*(1-xfit**2/a**2))
yfitlow = -np.sqrt(b**2*(1-xfit**2/a**2))
fig = plt.figure()
ax1 = plt.subplot2grid((1,1),(0,0))
ax1.plot(xmat,ymat,marker='.',ls='none',mew=1.0,mfc='none',mec='r',ms=1)
ax1.plot(xfit,yfithigh,lw=1.5,ls='-',c='k')
ax1.plot(xfit,yfitlow,lw=1.5,ls='-',c='k')
ax1.set_xlabel('$x$ (mm)',fontsize=lsize)
ax1.set_ylabel('$y$ (mm)',fontsize=lsize)
ax1.set_aspect('equal')
ax1.grid(b=True, which='major', color='k', linestyle=':')
fig.set_size_inches(figx,figy)
fig.savefig(pathDIR + name + '_xy_fit_ellipse.pdf',dpi=100,bbox_inches='tight')
#-------------------------------------------------------------------------------
# PLOT CIRCLE BASED ON requiv
theta = np.arange(0,2.0*np.pi,np.pi/100.0)
x = requiv*np.sin(theta)
y = requiv*np.cos(theta)
fig = plt.figure()
ax1 = plt.subplot2grid((1,1),(0,0))
ax1.plot(xmat,ymat,marker='.',ls='none',mew=1.0,mfc='none',mec='r',ms=1)
ax1.plot(x,y,lw=1.5,ls='-',c='k')
ax1.set_xlabel('$x$ (mm)',fontsize=lsize)
ax1.set_ylabel('$y$ (mm)',fontsize=lsize)
ax1.set_aspect('equal')
ax1.grid(b=True, which='major', color='k', linestyle=':')
fig.set_size_inches(figx,figy)
fig.savefig(pathDIR + name + '_xy_fit_sphere.pdf',dpi=100,bbox_inches='tight')
#-------------------------------------------------------------------------------
# CALCULATE THE ERROR IN THE FIT
error = (r-requiv)/r*100
print 'r = ' + str(requiv) + ' mm'
print 'error = ' + str(error) + '%'
#-------------------------------------------------------------------------------
