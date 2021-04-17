# Lab 2
#importing libraries
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy as sy
import matplotlib as matplotlib

#collected data from table for neon red light

red = np.loadtxt(open('groupcred.txt'), skiprows = 0)
xred = red[:,0]
yred = red[:,1]
sigma = np.ones(len(xred))
plt.figure()
plt.errorbar(xred,yred, yerr= sigma, c='red')
#add labels
plt.xlabel('Pixels',fontsize=10)
plt.ylabel('Spectra',fontsize=10)
plt.title('Neon Collected Data and Peaks Spectrometer',fontsize=12, y=1.02)


from scipy.signal import find_peaks

#plotting the peaks of the collected data of neon light
(peaks, ypeaks) = find_peaks(yred, height =5000)
print(peaks,ypeaks)
#copy and paste the peaks of the intensities and their corresponding pixels
neon = [1224 ,1282,  1379, 1409, 1490, 1536, 1568, 1582 ,1653, 1773]
ynearray = [ 9713.22,  6111.14,7166.72, 10895.19,  5315.36, 9034.47,  7059.06, 21055.46,  9359.8 ,  6389.66]
#plot the pixels against the intensities only using peaks to get an idea of the separation pattern between the points
plt.scatter(neon,ynearray)
plt.xlabel('Pixels')
plt.ylabel('Spectra')
plt.legend()
plt.show()

#Plot the table data
red_wavelength = [540,585,588,621,626,633,638,640,650,659]
relative_intensities_table = ([20000,20000,10000,10000,10000,10000,10000,20000,15000,10000])
#minimize the peaks of the table data in their length so that they match with the collected data(remove small peaks)

#matrix of Square fit
#define the arrays
xvalue = np.array(neon)
yvalue = red_wavelength
n = len(red_wavelength)

#develop the first part of the matrix
m_xN = np.array([[np.sum(xvalue**2),np.sum(xvalue)],[np.sum(xvalue**2),n]])
m_y = np.array([[np.sum(xvalue*yvalue)],[np.sum(yvalue)]])
#develop the code for the adjugate and the determinant of the matrix
determinant = (np.sum(xvalue**2)*n)-(np.sum((xvalue))**2)
adjugate = np.array([[n,-(np.sum(xvalue))],[-(np.sum(xvalue)),np.sum(xvalue**2)]])
f = (1/determinant)*adjugate
#develop the matrix identity
matrix_identity = np.dot(f, m_xN)
m_c = np.dot(f, m_y)
#separate the variables to gradient and y-intercept
m = m_c[0]
c = m_c[1]
print("gradient is",m,"y-intercept is",c)

#linear regression
#create the linear formula y=mx+c using previously established m and c
fit = m*neon + c

#plot the pixels against the y value of the regression
plt.plot(neon,fit, c='blue')
#plot the original data
plt.plot(neon,red_wavelength,'o',c='red')
plt.xlabel('Pixels',fontsize=10)
plt.ylabel('Wavelength',fontsize=10)
plt.title('Neon Data and Line of Best Fit Spectrometer',fontsize=12, y=1.02)
plt.legend()
plt.show()
#error propagation setup
#finding the standard deviation
xdata = np.array(neon)
ydata =np.array(red_wavelength)
ndata = len(xdata)
#applying equation
real_std = np.sqrt((1/(ndata-2))*(np.sum((ydata-((m*xdata)+c))**2)))
print("standard deviation is ",real_std)
#finding standard deviation for m
m_std = np.sqrt(ndata*(real_std**2))/(((np.sum(xdata**2))*ndata)-((np.sum(xdata))**2))
print("Standard deviation for m is", m_std)
#finding standard deviation for c
c_std = np.sqrt(((real_std**2)*(np.sum(xdata**2)))/((ndata*(np.sum(xdata**2)))-((np.sum(xdata))**2)))
print("Standard deviation for c is", c_std)

redfit = m*xred + c
plt.plot(redfit, yred,c= 'red')
plt.ylabel('Intensity',fontsize=10)
plt.xlabel('wavelength',fontsize=10)
plt.title('Callibration Data for Neon Spectrometer',fontsize=12, y=1.02)
plt.legend()
plt.show()


#plotting the purple light data
purple = np.loadtxt(open('groupcpurple.txt'), skiprows = 0)
xpurple = purple[:,0]
ypurple = purple[:,1]
sigma = np.ones(len(xpurple))
plt.figure()
#label the graph
plt.errorbar(xpurple, ypurple, yerr= sigma,c='purple')
plt.xlabel('Pixels',fontsize=10)
plt.ylabel('Spectra',fontsize=10)
plt.title('Purple/Hydrogen Collected Data and Peaks', fontsize=12, y=1.02)
#find the peaks in the purple light data
(purplepeaks, ypurplepeaks) = find_peaks(ypurple, height =5000)
print(purplepeaks, ypurplepeaks)
#make arrays for the purple peak data
p_peaks = ([226,352,637,1690])
py_peaks = ([6766.49,40711.28,65535., 65535.])
#plot the data of the purple peaks
plt.scatter(p_peaks,py_peaks)
plt.legend()
plt.show()
#using the equation we got from the square fit, plot the wavelengths of the purple light
fit_p = m*p_peaks + c

purplefit = m*xpurple + c
plt.plot(purplefit,ypurple,c='purple')
plt.ylabel('Intensity',fontsize=10)
plt.xlabel('wavelength',fontsize=10)
plt.title('Purple Callibration',fontsize=12, y=1.02)
plt.legend()
plt.show()


#same process goes for bulb data
#plotting lamp collected data 
lamp = np.loadtxt(open('groupcbulb.txt'), skiprows = 0)
xbulb = lamp[:,0]
ybulb = lamp[:,1]
sigma = np.ones(len(xbulb))
plt.figure()
#label the graph
plt.errorbar(xbulb, ybulb, yerr= sigma,c='orange')
plt.xlabel('Pixels',fontsize=10)
plt.ylabel('Spectra',fontsize=10)
plt.title('Lamp Collected Data and Peaks',fontsize=12, y=1.02)
(lamppeaks, ylamppeaks) = find_peaks(ybulb, height =40800)
print(lamppeaks, ylamppeaks)
l_peaks = np.array([1315, 1317, 1342, 1344, 1360, 1362, 1377, 1379, 1382, 1391, 1400, 1414, 1422, 1459])
l_ypeaks = np.array([40835.33, 41223.86, 40814.26, 40917.25, 40978.1 , 40800.22, 41081.08, 41071.72, 41116.19, 40884.48, 40942.99, 40903.2 ,40935.97, 40942.99])
plt.scatter(l_peaks,l_ypeaks)
plt.legend()
plt.show()
#using the equation we got from the square fit, plot the wavelengths of the bulb
fit_l = m*l_peaks + c
plt.show()
fitbulb = xbulb*m + c
plt.plot(fitbulb,ybulb, c='orange')
plt.ylabel('Intensity',fontsize=10)
plt.xlabel('wavelength',fontsize=10)
plt.title('Lamp Callibration Graph',fontsize=12, y=1.02)
plt.legend()
plt.show()








#plotting Sun collected data 
sun = np.loadtxt(open('Sun.txt'),skiprows = 0)
xsun = sun[:,0]
ysun = sun[:,1]
sigma = np.ones(len(xsun))
plt.figure()
#label the graph
plt.errorbar(xsun, ysun, yerr= sigma,c='yellow')
plt.xlabel('Pixels',fontsize=10)
plt.ylabel('Spectra',fontsize=10)
plt.title('Sun Collected Data',fontsize=12, y=1.02)

(sunxpeaks,sunypeaks) = find_peaks(ysun, height = 15500)
print(sunxpeaks,sunypeaks)

sunxpeak = np.array([687,696,699,702,713,748,750,753,755,768,773,775,782,790,792,795,797,803,806,831,838,841,854,863,882,884,893,895,910])
sunypeak = np.array([15667.31, 15752.74, 15888.63, 16033.98, 16019.89, 15981.69, 16019.89, 15676.72, 15523.74, 16198.47, 15887.28, 15901.27,
       16697.85, 15997.75, 16020.78, 15669.09, 15678.31, 16003.13,
       16344.66, 16035.95, 16118.15, 16213.12, 16396.95, 15990.73,
       15896.03, 15803.16, 15607.49, 15537.93, 15667.22])

plt.scatter(sunxpeak,sunypeak)
plt.legend()
plt.show()
#using the equation we got from the square fit, plot the wavelengths of the bulb
fit_s = m*sunxpeak + c


fitsun = m*xsun + c
plt.plot(fitsun,ysun,c='yellow')
plt.ylabel('Intensity',fontsize=10)
plt.xlabel('wavelength',fontsize=10)
plt.title('Sun Callibration Graph',fontsize=12)
plt.legend()
plt.show()





#plot the all sets together to see the pattern with the linear regression line
plt.plot(neon,red_wavelength,'bo',label = 'Neon',c='red')
plt.plot(p_peaks,fit_p,'bo',label = 'Hydrogen' ,c='purple')
plt.plot(p_peaks,fit_p,label ='Best Fit of Neon', c='blue')
plt.plot(neon,fit, c='blue')
plt.plot(l_peaks,fit_l,'bo',label ='Light Bulb', c='orange')
plt.plot(sunxpeak,fit_s,'bo',label ='Sun',c='yellow')
plt.xlabel('Pixels',fontsize=10)
plt.ylabel('Wavelength in nm',fontsize=10)
plt.title('Callibration Graph Based on Neon Spectrometer' ,fontsize=12, y=1.02)
plt.legend(loc='best')
plt.legend()
plt.show()

#viewing the Spectral Image from CCD
from astropy.io import fits
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
from astropy.utils.data import get_pkg_data_filename

#raw data for neon
neonimage = get_pkg_data_filename('neon.fit')
#opening the CCD image
newlist = fits.open(neonimage)
#finding the shape and type of the CCD image and the data collected
fits.info(neonimage)
neonimage_d = fits.getdata(neonimage, ext=0)
print(type(neonimage_d))
print(neonimage_d.shape)
plt.figure(11,figsize=(15,15)) #shape of CCD
newlist.close()
plt.imshow(neonimage_d, cmap='gray', vmin = 1000,vmax=1700) #plotting with high brightness to viw image
plt.title('CCD image of Neon Spectra')
plt.colorbar()
plt.legend()
plt.show()
#taking data from dark background image at 120s exposure (same steps as neon)
darkimage = get_pkg_data_filename('Dark120.fit')
darknewlist = fits.open(darkimage)
fits.info(darkimage)
darkimage_d = fits.getdata(darkimage, ext=0)
print(type(darkimage_d))
print(darkimage_d.shape)
plt.figure(11,figsize=(15,15))
darknewlist.close()
plt.imshow(darkimage_d, cmap='gray', vmin = 1000,vmax=1700)
plt.colorbar()
plt.show()

#taking slice from neon data 
plt.figure()
#251 is measured on DS9 to be the y point of highest intensity
#color red is for neon data
plt.plot(neonimage_d[251,:],color='red')
neond = neonimage_d[251,:] #plotting slice
plt.xlabel('Pixels',fontsize=10)
plt.ylabel('ADU',fontsize=10)
plt.title('Neon Light Data with Background for Telescope',fontsize=12)
plt.legend()
plt.show()
#taking slice from dark data 
plt.figure()
plt.plot(darkimage_d[251,:],c='black')
darkd = darkimage_d[251,:]
plt.xlabel('Pixels',fontsize=12)
plt.ylabel('ADU',fontsize=10)
plt.title('Background Dark Data 120s',fontsize=12)
plt.legend()
plt.show()
#using numpy library to subtract the background noise from raw data
newneon=np.abs(np.subtract(darkd,neond,dtype=np.ndarray)) #use subtract function due to type being ndarray
plt.plot(newneon,color='red')
plt.title('Neon Graph Collected Data for Telescope', fontsize=12)
plt.xlabel('Pixel', fontsize=10)
plt.ylabel('Intensity', fontsize=10)
plt.legend()
plt.show()
#tele stands for telescope neon data 
#finding peaks
(xpeaksneon_tele, ypeaksneon_tele) = find_peaks(newneon, height = 158)
print(xpeaksneon_tele, ypeaksneon_tele)
#manual of telescope wavelengths of most intense spectra
#using Ocean Optics to get values of the pixel in wavelength
table_2 = np.array(([ 39,  63, 113, 122, 140, 142, 156, 162, 187, 191, 203, 209, 218, 230, 243, 247, 258, 263,274, 294, 302, 308, 315],
                    [576,579,585,588,594,597,602,607,609,614,621,626,626.5,630,633,638,640,650,659,667,671,692,703]))
pix_t2 = table_2[0] #pixels
wav_t2 = table_2[1] #wavelengths callibrated

#linear least square fitting
xt = pix_t2
yt = wav_t2
nxt = len(xt)   #number of pixel points
print(nxt)
#construction of matrices
m_xNt = np.array([[np.sum(xt**2),np.sum(xt)],[np.sum(xt),nxt]])
m_yt = np.array([[np.sum(xt*yt)],[np.sum(yt)]])

#invert matrix m_xNt
dett = (np.sum(xt**2)*nxt) - ((np.sum(xt))**2)
adjugatet = np.array([[nxt,-(np.sum(xt))],[-(np.sum(xt)),np.sum(xt**2)]])
inverset = (1/dett) * adjugatet

#Multiply both matrices by inverse
m_identt = np.dot(inverset,m_xNt)
m_ct = np.dot(inverset,m_yt)

mt = m_ct[0]    #gradient
ct = m_ct[1]    #y-intercept
y_fit = mt*xt + ct
print('slope = ',mt)
print('intercept = ',ct)


#error propagation setup
#finding the standard deviation
xdata_t = np.array(pix_t2)
ydata_t =np.array(wav_t2)
ndata_t = len(xdata_t)
#applying equation
real_std_t = np.sqrt((1/(ndata_t-2))*(np.sum((ydata_t-((mt*xdata_t)+ct))**2)))
print("standard deviation is ",real_std_t)
#finding standard deviation for m
m_std_t = np.sqrt(ndata_t*(real_std_t**2))/(((np.sum(xdata_t**2))*ndata_t)-((np.sum(xdata_t))**2))
print("Standard deviation for m is", m_std_t)
#finding standard deviation for c
c_std_t = np.sqrt(((real_std_t**2)*(np.sum(xdata_t**2)))/((ndata_t*(np.sum(xdata_t**2)))-((np.sum(xdata_t))**2)))
print("Standard deviation for c is", c_std_t)

plt.title('Pixel to Wavelength', fontsize=12)
plt.xlabel('Pixel', fontsize=10)
plt.ylabel('Lambda (nm)', fontsize=10)
plt.plot(xt,y_fit)
plt.plot(xt[0:6],yt[0:6], 'bo',label='Neon from Ocean Optics Callibration for Telescope',c='red')
plt.legend(loc='best')
plt.legend()
plt.show()
#define the pixel range
arrayy = np.arange(0,765,1)
#get their wavelengths based on values of mt and ct
arrayywave = arrayy*mt + ct
plt.plot(arrayywave,newneon,color='red')
plt.title('Neon Callibration Graph for Telescope', fontsize=12)
plt.xlabel('Wavelength', fontsize=10)
plt.ylabel('Intensity', fontsize=10)
plt.legend()
plt.show()

#find the shape of wavelengths to use it as a range for stars
print(type(arrayywave))
#plotting the blackbody
#import the constants from scipy.constants
import scipy.constants as spc
h= spc.Planck
kk= spc.Boltzmann
c= spc.c
#define Planck function
def Planck(lambdaa, temp):
    aa = 2*h*(c**2)
    bb = h*c/(lambdaa*kk*temp)
    intensityofplanck = aa/((lambdaa**5) * ((np.exp(bb))-1))
    return intensityofplanck
#the_wavelengths were retrieved from the y=mt*arrayy+ct where y is the wavelengths and arrayy is the range of pixels
the_wavelength = np.arange(538e-3,870e-3,0.433987e-3)
#temperature of lamp is given as 3200 K
intensitynew = Planck(the_wavelength,3200)
plt.plot(the_wavelength*1e3,intensitynew,'g-',c='black')
plt.title('Blackbody',fontsize=12)
plt.xlabel('Wavelength in nm',fontsize=10)
plt.ylabel('Intensity',fontsize=10)
plt.legend()
plt.show()

#Flat Dark Data import
flatdark = get_pkg_data_filename('Flat-001D600s.fit')
flatdarklist = fits.open(flatdark)
fits.info(flatdark)
flatdark_d = fits.getdata(flatdark, ext=0)
print(type(flatdark_d))
print(flatdark_d.shape)
plt.figure(11,figsize=(15,15))
flatdarklist.close()
plt.imshow(flatdark_d, cmap='gray', vmin = 1000,vmax=1700)
plt.colorbar()
plt.show()
plt.figure()
plt.plot(flatdark_d[240,:],c='black')
flatdarkd = flatdark_d[240,:]
plt.xlabel('Pixels',fontsize=10)
plt.ylabel('ADU',fontsize=10)
plt.title('Flat Dark Data',fontsize=12, y=1.02)
plt.show()

#Flat light Data import
flatflat = get_pkg_data_filename('Flat-001F600s.fit')
flatflatlist = fits.open(flatflat)
fits.info(flatflat)
flatflat_d = fits.getdata(flatflat, ext=0)
print(type(flatflat_d))
print(flatflat_d.shape)
plt.figure(11,figsize=(15,15))
flatflatlist.close()
plt.imshow(flatflat_d, cmap='gray', vmin = 1000,vmax=1700)
plt.colorbar()
plt.show()
plt.figure()
plt.plot(flatflat_d[240,:],c='black')
flatflatd = flatflat_d[240,:]
plt.xlabel('Pixels',fontsize=10)
plt.ylabel('ADU',fontsize=10)
plt.title('Flat Light Data of Lamp',fontsize=12, y=1.02)
plt.legend()
plt.show()
 
#taking data from enif star
enifimage = get_pkg_data_filename('Enif.fit')
enifnewlist = fits.open(enifimage)
fits.info(enifimage)
enifimage_d = fits.getdata(enifimage, ext=0)
print(type(enifimage_d))
print(enifimage_d.shape)
plt.figure(11,figsize=(15,15))
enifnewlist.close()
plt.imshow(enifimage_d, cmap='gray', vmin = 1000,vmax=1700)
plt.colorbar()
plt.show()

plt.figure(figsize=(10,3))
plt.plot(enifimage_d[330,:],c='black')
plt.xlabel('Pixels',fontsize=10)
plt.ylabel('ADU',fontsize=10)
plt.title('Enif Data with Background',fontsize=12, y=1.02)
plt.legend()
plt.show()
#raw data minus dark data over lamp data minus dark data times the Planck function
enifd = enifimage_d[330,:]
darkdd = darkimage_d[330,:]
numert = (np.abs(np.subtract(enifd,darkdd,dtype=np.ndarray)))
denomt = (np.abs(np.subtract(flatflatd,darkdd,dtype=np.ndarray)))
fractionpart = ((np.divide(numert,denomt,dtype=np.ndarray))*intensitynew)
fractionpart = np.array(fractionpart.tolist())
#plot against wavelength taken from previous y=mt*arrayy+ct
plt.figure(figsize=(10,3))
plt.plot(the_wavelength,fractionpart,c='black')
plt.title('Enift Star Callibration',fontsize=12)
plt.xlabel('Wavelength in micron',fontsize=10)
plt.ylabel('Intensity',fontsize=10)
plt.legend()
plt.show()
#taking data from Vega star
vegaimage = get_pkg_data_filename('Vega.fit')
veganewlist = fits.open(vegaimage)
fits.info(vegaimage)
vegaimage_d = fits.getdata(vegaimage, ext=0)
plt.figure(11,figsize=(15,15))
veganewlist.close()
plt.imshow(vegaimage_d, cmap='gray', vmin = 1000,vmax=1700)
plt.colorbar()
plt.show()
plt.figure()
plt.plot(vegaimage_d[345,:],c='black')
plt.xlabel('Pixels',fontsize=10)
plt.ylabel('ADU',fontsize=10)
plt.title('Vega Light Data with Background',fontsize=12, y=1.02)
plt.show()
vegad = vegaimage_d[345,:]
darkvega = darkimage_d[345,:]
numertvega = (np.abs(np.subtract(vegad,darkvega,dtype=np.ndarray)))
fractionpartvega = ((np.divide(numertvega,denomt,dtype=np.ndarray))*intensitynew)
fractionpartvega = np.array(fractionpartvega.tolist())
plt.figure(figsize=(10,3))
plt.plot(the_wavelength,fractionpartvega,c='black')
plt.title('Vega Star Callibration',fontsize=12)
plt.xlabel('Wavelength in micron',fontsize=10)
plt.ylabel('Intensity',fontsize=10)
plt.legend()
plt.show()
#taking data from Navi star
naviimage = get_pkg_data_filename('Navi.fit')
navinewlist = fits.open(naviimage)
fits.info(naviimage)
naviimage_d = fits.getdata(naviimage, ext=0)
plt.figure(11,figsize=(15,15))
navinewlist.close()
plt.imshow(naviimage_d, cmap='gray', vmin = 1000,vmax=1700)
plt.colorbar()
plt.show()
plt.figure()
plt.plot(naviimage_d[349,:],c='black')
plt.xlabel('Pixels',fontsize=10)
plt.ylabel('ADU',fontsize=10)
plt.title('Navi Star Data with Background',fontsize=12, y=1.02)
plt.show()
navid = naviimage_d[349,:]
darknavi = darkimage_d[349,:]
numertnavi = (np.abs(np.subtract(navid,darknavi,dtype=np.ndarray)))
fractionpartnavi = ((np.divide(numertnavi,denomt,dtype=np.ndarray))*intensitynew)
fractionpartnavi = np.array(fractionpartnavi.tolist())
plt.figure(figsize=(10,3))
plt.plot(the_wavelength,fractionpartnavi,c='black')
plt.title('Navi Star Callibration',fontsize=12)
plt.xlabel('Wavelength in micron',fontsize=10)
plt.ylabel('Intensity',fontsize=10)
plt.legend()
plt.show()


#taking data from Scheat star
scheatimage = get_pkg_data_filename('scheat.fit')
scheatnewlist = fits.open(scheatimage)
fits.info(scheatimage)
scheatimage_d = fits.getdata(scheatimage, ext=0)
plt.figure(11,figsize=(15,15))
scheatnewlist.close()
plt.imshow(scheatimage_d, cmap='gray', vmin = 1000,vmax=1700)
plt.colorbar()
plt.show()
plt.figure()
plt.plot(scheatimage_d[349,:],c='black')
plt.xlabel('Pixels',fontsize=10)
plt.ylabel('ADU',fontsize=10)
plt.title('Scheat Star Data with Background',fontsize=12, y=1.02)
plt.show()
scheatd = scheatimage_d[349,:]
darkscheat = darkimage_d[349,:]
numertscheat = (np.abs(np.subtract(scheatd,darkscheat,dtype=np.ndarray)))
fractionpartscheat = ((np.divide(numertscheat,denomt,dtype=np.ndarray))*intensitynew)
fractionpartscheat = np.array(fractionpartscheat.tolist())
plt.figure(figsize=(10,3))
plt.plot(the_wavelength,fractionpartscheat,c='black')
plt.title('Scheat Star Callibration',fontsize=12)
plt.xlabel('Wavelength in micron',fontsize=10)
plt.ylabel('Intensity',fontsize=10)
plt.legend()
plt.show()


#taking data from Neptune Planet
neptuneimage = get_pkg_data_filename('Neptune.fit')
#import data from dark exposure of 300s as recorded from telescope
dark300 = get_pkg_data_filename('dark300.fit')
dark300image_d = fits.getdata(dark300, ext=0)
dark300d = dark300image_d[349,:]
neptunenewlist = fits.open(neptuneimage)
fits.info(neptuneimage)
neptuneimage_d = fits.getdata(neptuneimage, ext=0)
plt.figure(11,figsize=(15,15))
neptunenewlist.close()
plt.imshow(neptuneimage_d, cmap='gray', vmin = 1000,vmax=1700)
plt.colorbar()
plt.show()
plt.figure()
#slice at highest intensity point
plt.plot(neptuneimage_d[349,:],c='black')
plt.xlabel('Pixels',fontsize=10)
plt.ylabel('ADU',fontsize=10)
plt.title('Neptune Data with Background',fontsize=12, y=1.02)
plt.show()
neptuned = neptuneimage_d[349,:]
numertneptune = (np.abs(np.subtract(neptuned,dark300d,dtype=np.ndarray)))
denomtneptune = (np.abs(np.subtract(flatflatd,dark300d,dtype=np.ndarray)))
fractionpartneptune = ((np.divide(numertneptune,denomtneptune,dtype=np.ndarray))*intensitynew)
fractionpartneptune = np.array(fractionpartneptune.tolist())
plt.figure(figsize=(10,3))
plt.plot(the_wavelength,fractionpartneptune,c='black')
plt.title('Neptune Callibration',fontsize=12)
plt.xlabel('Wavelength in micron',fontsize=10)
plt.ylabel('Intensity',fontsize=10)
plt.legend()
plt.show()

#finding the temperature of the stars and Neptune
#the sun
b = 2.897771955*(10**(-3)) #meter kelvin
maximum_sun = np.where(np.max(ysun)==ysun)
maximum_sun_wave = fitsun[maximum_sun[0][0]]
#wein's law
temperature_sun = b/(maximum_sun_wave*(1e-9)) #since data was in nm
print(temperature_sun)

